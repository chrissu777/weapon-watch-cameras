import time
import threading
import queue
import cv2
import os

from detect import detect_worker
from record import record_worker
from track import track_worker

import firebase_admin
from firebase_admin import credentials, firestore

ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_name, q_detect, q_record, q_track, q_display, school, shutdown_flag=None):
    global ACTIVE_EVENT

    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })

    db = firestore.client()
    ref = db.collection('schools').document(school)

    def on_snapshot(docs, changes, ts):
        ACTIVE_EVENT = docs[0].to_dict().get('Active Event', False)

    watch = ref.on_snapshot(on_snapshot)

    cap = cv2.VideoCapture(rtsp_url)
    
    # Set RTSP connection parameters
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request specific FPS
    
    # Try to read a test frame to verify connection
    connection_attempts = 0
    max_attempts = 3
    
    while connection_attempts < max_attempts:
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"\n[INFO] {cam_name}: Successfully connected to RTSP stream")
                break
            else:
                print(f"[WARNING] {cam_name}: Connected but no frames received (attempt {connection_attempts + 1}/{max_attempts})")
        else:
            print(f"\n[WARNING] {cam_name}: Cannot open RTSP stream (attempt {connection_attempts + 1}/{max_attempts})")
        
        connection_attempts += 1
        cap.release()
        time.sleep(3)  # Wait before retry
        cap = cv2.VideoCapture(rtsp_url)
        # Reapply settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    if connection_attempts >= max_attempts:
        print(f"[ERROR] {cam_name}: Failed to connect to RTSP stream after {max_attempts} attempts")
        print(f"[INFO] {cam_name}: Check network connectivity and RTSP URL: {rtsp_url}\n")
        cap.release()
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)    
    print(f"[INFO] {cam_name}: Live stream at {fps:.1f} FPS")

    try:
        while True:
            if shutdown_flag and shutdown_flag.is_set():
                break
                
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] {cam_name}: Failed to read frame")
                break

            try:
                q_detect.put(frame.copy(), timeout=0.1)
            except queue.Full:
                print(f'[WARNING] Detection queue full for {cam_name}')
                pass
            
            try:
                q_record.put(frame, timeout=0.1)
            except queue.Full:
                print(f'[WARNING] Recording queue full for {cam_name}')
                pass
                
            if ACTIVE_EVENT:
                try:
                    q_track.put(frame, timeout=0.1)
                except queue.Full:
                    print(f'[WARNING] Tracking queue full for {cam_name}')
                    pass
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n[INFO] {cam_name} received keyboard interrupt")
    finally:
        cap.release()
        
        # Signal end of video to all workers
        try:
            q_detect.put(None, timeout=1.0)  # Sentinel value to signal end
        except queue.Full:
            pass
            
        try:
            q_record.put(None, timeout=1.0)  # Sentinel value to signal end
        except queue.Full:
            pass

        try:
            q_display.put(None, timeout=1.0)  # Sentinel value to signal end
        except queue.Full:
            pass

def threaded_process(rtsp_url, cam_id, cam_name, school, infer_weapon, yolo, reid_model, reid_transform, output_dir, shutdown_flag=None, q_display=None):
    q_detect = queue.Queue(maxsize=64)
    q_record = queue.Queue(maxsize=64) 
    q_track = queue.Queue(maxsize=64)
    
    if q_display is None:
        q_display = queue.Queue(maxsize=32)

    t_read = threading.Thread(
        target=frame_reader,
        args=(rtsp_url, cam_name, q_detect, q_record, q_track, q_display, school, shutdown_flag),
        name=f"{cam_name}-reader"
    )
    t_detect = threading.Thread(
        target=detect_worker,
        args=(q_detect, q_display, cam_id, cam_name, school, infer_weapon, output_dir, shutdown_flag),
        name=f"{cam_name}-detector"
    )
    t_record = threading.Thread(
        target=record_worker,
        args=(q_record, cam_id, cam_name),
        name=f"{cam_name}-recorder"
    )
    t_track = threading.Thread(
        target=track_worker,
        args=(q_track, cam_id, school, yolo.model, reid_model, reid_transform),
        name=f"{cam_name}-tracker"
    )

    t_read.start()
    t_detect.start()
    t_record.start()
    t_track.start()

    t_read.join()
    t_detect.join()
    t_record.join()
    t_track.join()
