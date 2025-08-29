import os
import threading
import time
import signal
import shutil
import queue

from process import threaded_process
from load_onnx import load_onnx
from gui_display import gui_display_worker

# Global shutdown flag
shutdown_flag = threading.Event()

import torch
# import torchreid
from torchvision import transforms
from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore

def signal_handler(sig, frame):
    print('\n\n[INFO] Ctrl+C pressed. Shutting down...')
    shutdown_flag.set()

if __name__ == '__main__':
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")

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
    cams = db.collection('schools').document('UMD').collection('cameras').stream()

    # Load shared detection model
    print("\n[INFO] Loading ONNX detection model...")
    infer_weapon = load_onnx('models/detectionmodel.onnx')    
    print("[INFO] Detection model loaded and ready")

    # Load shared YOLO model
    print("\n[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")
    yolo.fuse()
    print("[INFO] YOLO model loaded.")

    # Load shared ReID model
    # print("\n[INFO] Loading ReID model...")
    # reid_model = torchreid.models.build_model(
    #     name='osnet_ibn_x1_0',
    #     num_classes=1000,
    #     loss='softmax',
    #     pretrained=True
    # )
    # device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # reid_model.to(device).eval()
    # print(f"[INFO] ReID model loaded on {device}\n")

    # reid_transform = transforms.Compose([
    #     transforms.Resize((256, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    output_dir = 'testing/outputs/detected'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"[INFO] Output directory: {output_dir}")
        
    print("\n[INFO] Starting video processing...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    threads = []
    gui_queues = {}  # Dictionary to store GUI queues by camera ID
    
    for cam in cams:
        cam_id = cam.id
        cam_name = cam.to_dict()['name']
        rtsp_url = cam.to_dict()['video link']
                
        reid_model = 0
        reid_transform = 0

        # Create GUI queue for this camera
        gui_queues[cam_id] = queue.Queue(maxsize=32)

        t = threading.Thread(
            target=threaded_process,
            args=(rtsp_url, cam_id, cam_name, 'UMD', infer_weapon, yolo, reid_model, reid_transform, output_dir, shutdown_flag, gui_queues[cam_id]),
            name=f"{cam_name}-main-thread",
            daemon=False  # Don't kill threads on main exit - let them finish naturally
        )
        threads.append(t)
        t.start() 
        
    # Start GUI display thread
    print("[INFO] Starting GUI display thread...")
    gui_thread = threading.Thread(
        target=gui_display_worker,
        args=(gui_queues, shutdown_flag),
        name="GUI-Display",
        daemon=False
    )
    threads.append(gui_thread)
    gui_thread.start()
    print("[INFO] GUI window opened. Press 'q' or ESC in the GUI window to stop all processing.")
        
    try:
        # Wait for all threads to complete naturally, but check for Ctrl+C periodically
        while any(t.is_alive() for t in threads) and not shutdown_flag.is_set():
            time.sleep(0.1)  # Check every 100ms
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
        shutdown_flag.set()
    
    if not shutdown_flag.is_set():
        # All videos completed naturally
        print("\n[INFO] All videos processed successfully!")
        print("[INFO] Press Ctrl+C to exit program")
        # Wait for user to press Ctrl+C
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")
            shutdown_flag.set()
    
    print("\n[INFO] Waiting for threads to complete...")
    for t in threads:
        if t.is_alive():
            t.join(timeout=3.0)  # Wait max 3 seconds per thread
    
    print("[INFO] Program terminated successfully")
    
    # Force exit to avoid hanging on C++ cleanup issues
    print("[INFO] Forcing program exit...")
    import os
    os._exit(0)