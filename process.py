import time
import threading
import queue
import cv2
import os


from detect import detect_worker
from record import record_worker
# from track import track_worker
from stream import RTSPStream

import firebase_admin
from firebase_admin import credentials, firestore

# Global flag to control shooter tracking logic
ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_name, q_detect, q_record, q_track, q_display, school, shutdown_flag=None):
    global ACTIVE_EVENT

    # if not firebase_admin._apps:
    #     cred = credentials.Certificate("serviceAccountKey.json")
    #     firebase_admin.initialize_app(cred, {
    #         "storageBucket": "weapon-watch.firebasestorage.app"
    #     })

    # db = firestore.client()
    # ref = db.collection('schools').document(school)

    # def on_snapshot(docs, changes, ts):
    #     ACTIVE_EVENT = docs[0].to_dict().get('Active Event', False)
    #     # print(f"[{cam_name}] ACTIVE EVENT: {ACTIVE_EVENT}")

    # watch = ref.on_snapshot(on_snapshot)

    cap = cv2.VideoCapture(rtsp_url)
    
    # Set RTSP connection parameters
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    cap.set(cv2.CAP_PROP_FPS, 15)  # Request specific FPS
    
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
        cap.set(cv2.CAP_PROP_FPS, 15)
    
    if connection_attempts >= max_attempts:
        print(f"[ERROR] {cam_name}: Failed to connect to RTSP stream after {max_attempts} attempts")
        print(f"[INFO] {cam_name}: Check network connectivity and RTSP URL: {rtsp_url}")
        cap.release()
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)    
    print(f"[INFO] {cam_name}: Live stream at {fps:.1f} FPS")

    i = 0
    try:
        while True:
            # Check for shutdown signal
            if shutdown_flag and shutdown_flag.is_set():
                break
                
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] {cam_name}: Failed to read frame {i}")
                break

            # Send every frame to detection
            try:
                q_detect.put(frame.copy(), timeout=0.1)
            except queue.Full:
                print(f'[WARNING] Detection queue full for {cam_name}')
                pass  # Skip detection if queue is full
            
            try:
                q_record.put(frame, timeout=0.05)
            except queue.Full:
                # Skip recording frame if queue is full
                pass
                
            if ACTIVE_EVENT:
                try:
                    q_track.put(frame, timeout=0.05)
                except queue.Full:
                    pass

            i += 1
            
            # Use actual video frame rate for timing (for live streams, can be minimal)
            time.sleep(0.01)  # Minimal delay for live feeds
            
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


    # stream = RTSPStream(rtsp_url)
    # INVALID_FRAME_COUNT = 0

    # while True:
    #     frame = stream.read()
    #     if frame is not None:
    #         q_detect.put(frame)
    #         q_record.put(frame)
    #         if ACTIVE_EVENT:
    #             q_track.put(frame)
    #         INVALID_FRAME_COUNT = 0
    #     else:
    #         print(f"[{cam_name}] Invalid frame received.")
    #         INVALID_FRAME_COUNT += 1
    #         time.sleep(0.1)
    #         if INVALID_FRAME_COUNT >= 10:
    #             break``
    #     time.sleep(0.2)

    # print(f"\n[{cam_name}] Too many invalid frames. Stopping stream.\n")
    # stream.stop()
    # watch.unsubscribe()

def threaded_process(rtsp_url, cam_id, cam_name, school, infer_weapon, yolo, reid_model, reid_transform, output_dir, shutdown_flag=None, q_display=None):
    # Thread-safe queues with larger buffers
    q_detect = queue.Queue(maxsize=64)
    q_record = queue.Queue(maxsize=64) 
    q_track = queue.Queue(maxsize=64)
    if q_display is None:
        q_display = queue.Queue(maxsize=32)

    # Create threads
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
    # t_track = threading.Thread(
    #     target=track_worker,
    #     args=(q_track, cam_id, school, yolo.model, reid_model, reid_transform),
    #     name=f"{cam_name}-tracker"
    # )

    # Start threads
    t_read.start()
    t_detect.start()
    t_record.start()
    # t_track.start()

    # Join threads
    t_read.join()
    t_detect.join()
    t_record.join()
    # t_track.join()
