import time
import threading
import queue
import cv2
import os

from tqdm import tqdm

from detect import detect_worker
from record import record_worker
# from track import track_worker
from stream import RTSPStream

import firebase_admin
from firebase_admin import credentials, firestore

# Global flag to control shooter tracking logic
ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_name, q_detect, q_record, q_track, school, shutdown_flag=None):
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

    # print(f"Playing: {os.path.basename(rtsp_url)}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open {rtsp_url}")
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = 1.0 / fps if fps > 0 else 0.033  # fallback to ~30fps
    
    # print(f"[INFO] {cam_name}: {total_frames} frames at {fps:.2f} FPS")
    pbar = tqdm(total=total_frames, desc=f"{os.path.basename(rtsp_url)} ({fps:.1f}fps)", unit='frame')

    i = 0
    try:
        while True:
            # Check for shutdown signal
            if shutdown_flag and shutdown_flag.is_set():
                print(f"\n[INFO] {cam_name} frame reader shutting down...")
                break
                
            ret, frame = cap.read()
            if not ret:
                break

            # Send every other frame to detection (every 2nd frame)
            if i % 2 == 0:
                try:
                    q_detect.put(frame.copy(), timeout=0.1)
                except queue.Full:
                    print('[WARNING] Detection queue full')
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
            pbar.update(1)
            
            # Use actual video frame rate for timing
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        print(f"\n[INFO] {cam_name} received keyboard interrupt")
    finally:
        pbar.close()
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
            
        print(f"[INFO] {cam_name} finished processing video")


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
    #             break
    #     time.sleep(0.2)

    # print(f"\n[{cam_name}] Too many invalid frames. Stopping stream.\n")
    # stream.stop()
    # watch.unsubscribe()

def threaded_process(rtsp_url, cam_id, cam_name, school, infer_weapon, yolo, reid_model, reid_transform, i, output_dir, shutdown_flag=None):
    # Thread-safe queues
    q_detect = queue.Queue(maxsize=32)
    q_record = queue.Queue(maxsize=32)
    q_track = queue.Queue(maxsize=32)

    # Create threads
    t_read = threading.Thread(
        target=frame_reader,
        args=(rtsp_url, cam_name, q_detect, q_record, q_track, school, shutdown_flag),
        name=f"{cam_name}-reader"
    )
    t_detect = threading.Thread(
        target=detect_worker,
        args=(q_detect, cam_id, cam_name, school, infer_weapon, i, output_dir, shutdown_flag),
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
