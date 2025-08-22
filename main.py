import os
import threading
import cv2
import queue

from process import threaded_process

import torch
import torchreid
from torchvision import transforms
from ultralytics import YOLO

import tensorflow as tf

import firebase_admin
from firebase_admin import credentials, firestore

def display_loop(q_display):
    print("\n[INFO] starting display\n")
    while True:
        try:
            print("yo")
            cam_name, frame = q_display.get()
            print(frame)
            cv2.imshow(cam_name, frame)
            print("gurt")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except queue.Empty:
            continue
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Suppress logs
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"
    # Force TCP transport for RTSP
    #os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    # Firebase initialization
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })
    db = firestore.client()

    # Load shared detection model
    print("\n[INFO] Loading detection model...")
    detection_model = tf.saved_model.load("detectionmodel")
    infer_weapon = detection_model.signatures['serving_default']
    print("[INFO] detection model loaded")
    
    # Load shared YOLO model
    print("\n[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")
    yolo.fuse()
    print("[INFO] YOLO model loaded.")

    # Load shared ReID model
    print("\n[INFO] Loading ReID model...")
    reid_model = torchreid.models.build_model(
        name='osnet_ibn_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    reid_model.to(device).eval()
    print(f"[INFO] ReID model loaded on {device}\n")

    reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Fetch cameras
    cams = db.collection('schools').document('UMD').collection('cameras').stream()

    threads = []
    q_display = queue.Queue(maxsize=32)
    for cam in cams:
        rtsp_url = cam.to_dict()['video_link']
        cam_id = cam.id
        cam_name = cam.to_dict()['name']
        

        if cam_name == "Camera 1":
            t = threading.Thread(
                target=threaded_process,
                args=(rtsp_url, cam_id, cam_name, 'UMD', infer_weapon, yolo, reid_model, reid_transform, q_display, db),
                name=f"{cam_name}-main-thread",
                daemon=True
            )
            threads.append(t)
            t.start() 
        
    display_loop(q_display)   

    for t in threads:
        t.join()