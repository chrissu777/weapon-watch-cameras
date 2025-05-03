import cv2
import os

from collections import deque
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from cloud import encrypt_upload

ACTIVE = False

def record_worker(q_record, cam_id, cam_name, buffer_size=100):
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
    s3_key = f"University_of_Maryland_College_Park*163286*{cam_id}*"
    save_file = "recordings/" + s3_key
    
    if not os.path.exists('recordings'):
        os.makedirs('recordings')
    
    try: 
        while True:
            frame = q_record.get()
            buf.append(frame)

            if ACTIVE and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                h, w = frame.shape[:2]
                
                save_file += str(datetime.now().strftime("%H:%M:%S")) + ".mp4"
                s3_key += str(datetime.now().strftime("%H:%M:%S")) + ".mp4"
                
                writer = cv2.VideoWriter(save_file, fourcc, 5.0, (w, h))
                for f in buf:
                    writer.write(f)
                buf.clear()
                
                formatted_time = datetime.now().strftime("%H:%M:%S")
                print(f"RECORDING STARTED AT {formatted_time} FOR {cam_name}")

            if ACTIVE and writer is not None:
                writer.write(frame)

            if not ACTIVE and writer is not None:                
                writer.release()
                writer = None

                print("ATTEMPTING TO ENCRYPT AND UPLOAD")

                encrypt_upload.encrypt_and_upload(save_file, s3_key, cam_name)
                
    except KeyboardInterrupt:
        watch.unsubscribe()