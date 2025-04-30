import cv2
from collections import deque
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from cloud import encrypt_upload

import time

ACTIVE = False

def record_worker(q_record, cam_id, buffer_size=100):
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    # listen for the “Active Event” flag in Firestore:
    def on_snapshot(docs, changes, ts):
        global ACTIVE
        ACTIVE = docs[0].to_dict().get('Active Event', False)

    ref = db.collection('schools').document('UMD')
    watch = ref.on_snapshot(on_snapshot)

    buf = deque(maxlen=buffer_size)
    writer = None
    save_file = f"University of Maryland-College Park*163286*{cam_id}."

    try:
        while True:
            frame = q_record.get()
            buf.append(frame)

            if ACTIVE and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                save_file += time.time() + ".mp4"
                writer = cv2.VideoWriter(save_file, fourcc, 30.0, (w, h))
                for f in buf:
                    writer.write(f)
                buf.clear()
                formatted_time = datetime.now().strftime("%H:%M:%S")
                print(f"RECORDING STARTED AT {formatted_time} FOR CAM {cam_id}")

            if ACTIVE and writer is not None:
                writer.write(frame)

            if not ACTIVE and writer is not None:
                writer.release()
                writer = None
                
                formatted_time = datetime.now().strftime("%H:%M:%S")
                encrypt_upload.encrypt_and_upload(save_file, save_file)
                print(f"RECORDING SAVED AT {formatted_time} FOR CAM {cam_id}")
                
    except KeyboardInterrupt:
        watch.unsubscribe()