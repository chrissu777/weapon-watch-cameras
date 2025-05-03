import os
import threading
from process import threaded_process

import torch
import torchreid
from torchvision import transforms
from ultralytics import YOLO

import tensorflow as tf

import firebase_admin
from firebase_admin import credentials, firestore

if __name__ == '__main__':
    # Suppress logs
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"

    # Firebase initialization
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })
    db = firestore.client()

    # Load shared detection model
    print("[INFO] Loading detection model...")
    detection_model = tf.saved_model.load("detectionmodel")
    # Load shared YOLO model
    print("[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")
    yolo.fuse()
    print("[INFO] YOLO model loaded.")

    # Load shared ReID model
    print("[INFO] Loading ReID model...")
    reid_model = torchreid.models.build_model(
        name='osnet_ibn_x1_0',
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    reid_model.to(device).eval()
    print(f"[INFO] ReID model loaded on {device}")

    reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Fetch cameras
    cams = db.collection('schools').document('UMD').collection('cameras').stream()

    threads = []
    for cam in cams:
        cam_id = cam.id
        data = cam.to_dict()
        cam_name = data.get('name', f'Cam-{cam_id}')
        rtsp_url = data.get('video_link', '')
        if cam_name == "Camera 1":
            t = threading.Thread(
                target=threaded_process,
                args=(rtsp_url, cam_id, cam_name, 'UMD', detection_model, yolo, reid_model, reid_transform),
                name=f"{cam_name}-main-thread",
                daemon=True
            )
            threads.append(t)
            t.start()
        

    for t in threads:
        t.join()
