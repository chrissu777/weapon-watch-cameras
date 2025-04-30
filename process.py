import time
from multiprocessing import Process
from multiprocessing import Queue

from detect import detect_worker
from record import record_worker
from track import track_worker
from stream import RTSPStream

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_id, q_detect, q_record, q_track):
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })
    db = firestore.client()
    
    def on_snapshot(docs, changes, ts):
        global ACTIVE_EVENT
        ACTIVE_EVENT = docs[0].to_dict().get('Active Event', False)
        print(f"ACTIVE EVENT IS: {ACTIVE_EVENT}")
    
    ref = db.collection('schools').document('UMD')
    watch = ref.on_snapshot(on_snapshot)
    
    stream = RTSPStream(rtsp_url)
    INVALID_FRAME_COUNT = 0

    while True:
        frame = stream.read()
        
        if frame is not None:
            q_detect.put(frame)
            
            if ACTIVE_EVENT:
                q_record.put(frame)
                q_track.put(frame)
            
            INVALID_FRAME_COUNT = 0
        else:
            print("INVALID FRAME")
            INVALID_FRAME_COUNT += 1
            time.sleep(0.1)
            
            if INVALID_FRAME_COUNT == 10:
                break

    print(f"\nTOO MANY INVALID FRAMES: {cam_id} CAMERA STREAM ENDED\n")
    stream.stop()
    watch.unsubscribe()

def process(rtsp_url, cam_id, school):
    q_detect = Queue(maxsize=32)
    q_record = Queue(maxsize=32)
    q_track = Queue(maxsize=32)

    p_read = Process(target=frame_reader, args=(rtsp_url, cam_id, q_detect, q_record, q_track), name="reader")
    p_detect = Process(target=detect_worker, args=(q_detect, cam_id, school), name="detector")
    p_record = Process(target=record_worker, args=(q_record, cam_id), name="recorder")
    p_track = Process(target=track_worker, args=(q_track, cam_id, school), name="tracker")
    
    p_read.start()
    p_detect.start()
    p_record.start()
    p_track.start()
    
    p_read.join()
    p_detect.join()
    p_record.join()
    p_track.join()