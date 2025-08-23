import time
import threading
import queue
import cv2
import os

from tqdm import tqdm

from detect import detect_worker
from record import record_worker
from track import track_worker
from stream import RTSPStream

import firebase_admin
from firebase_admin import credentials, firestore

# Global flag to control shooter tracking logic
ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_name, q_detect, q_record, q_track, school):
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Error: Could not open {rtsp_url}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=os.path.basename(rtsp_url), unit='frame')

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if i%2 == 0:
            try:
                q_detect.put(frame)
                q_record.put(frame)
                if ACTIVE_EVENT:
                    q_track.put(frame)
            except queue.Full:
                pass  # drop frame instead of blocking
        i += 1
        pbar.update(1)
    cap.release()

def threaded_process(rtsp_url, cam_id, cam_name, school, infer_weapon, yolo, reid_model, reid_transform, q_display, db):
    # Thread-safe queues
    q_detect = queue.Queue(maxsize=32)
    q_record = queue.Queue(maxsize=32)
    q_track = queue.Queue(maxsize=32)

    # Create threads
    t_read = threading.Thread(
        target=frame_reader,
        args=(rtsp_url, cam_name, q_detect, q_record, q_track, school),
        name=f"{cam_name}-reader"
    )
    t_detect = threading.Thread(
        target=detect_worker,
        args=(q_detect, cam_id, cam_name, school, infer_weapon, q_display, db),
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

    # Start threads
    t_read.start()
    t_detect.start()
    t_record.start()
    t_track.start()

    # Join threads
    t_read.join()
    t_detect.join()
    t_record.join()
    t_track.join()
