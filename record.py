import cv2
import os
import time

from collections import deque
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from cloud import encrypt_upload

ACTIVE = False

def record_worker(q_record, cam_id, cam_name, buffer_minutes=3):
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # listen for the "Active Event" flag in Firestore:
    def on_snapshot(docs, changes, ts):
        global ACTIVE
        ACTIVE = docs[0].to_dict().get('Active Event', False)

    ref = db.collection('schools').document('UMD')
    watch = ref.on_snapshot(on_snapshot)

    # Dynamic buffer that adjusts based on measured frame rate
    rolling_buffer = deque()
    writer = None
    s3_key_base = f"UMD*163286*{cam_id}*{cam_name}*"
    
    # Track frame timing to calculate actual FPS
    frame_timestamps = deque(maxlen=30)  # Track last 30 frame times
    
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    try: 
        while True:
            frame = q_record.get()
            current_time = time.time()
            frame_timestamps.append(current_time)
            
            # Calculate actual FPS and adjust buffer size
            if len(frame_timestamps) > 10:
                time_diff = frame_timestamps[-1] - frame_timestamps[0]
                actual_fps = (len(frame_timestamps) - 1) / time_diff
                target_buffer_size = int(actual_fps * 60 * buffer_minutes)  # 3 minutes worth
                
                # Resize buffer if needed
                if rolling_buffer.maxlen != target_buffer_size:
                    new_buffer = deque(rolling_buffer, maxlen=target_buffer_size)
                    rolling_buffer = new_buffer
            
            rolling_buffer.append(frame)  # Always add to rolling buffer

            if ACTIVE and writer is None:
                # Start recording: write buffer + continue recording
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                
                # Use measured FPS or default to 15 if not enough data
                video_fps = actual_fps if len(frame_timestamps) > 10 else 30.0
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_key = s3_key_base + timestamp + ".mp4"
                save_file = "recordings/" + s3_key
                
                writer = cv2.VideoWriter(save_file, fourcc, video_fps, (w, h))
                
                # Write the 3-minute buffer first (pre-event footage)
                for buffered_frame in rolling_buffer:
                    writer.write(buffered_frame)
                
                formatted_time = datetime.now().strftime("%H:%M:%S")
                print(f"[INFO] {cam_name}: Recording started at {formatted_time} (including 3min pre-buffer)")

            if ACTIVE and writer is not None:
                writer.write(frame)

            if not ACTIVE and writer is not None:                
                writer.release()
                writer = None

                encrypt_upload.encrypt_and_upload(save_file, s3_key, cam_name)
                
    except KeyboardInterrupt:
        watch.unsubscribe()