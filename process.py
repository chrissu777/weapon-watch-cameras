import time
import threading
import queue

from detect import detect_worker
from record import record_worker
from track import track_worker
from stream import RTSPStream

import firebase_admin
from firebase_admin import credentials, firestore

# Global flag to control shooter tracking logic
ACTIVE_EVENT = False

def frame_reader(rtsp_url, cam_name, q_detect, q_record, q_track, school):
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
        # print(f"[{cam_name}] ACTIVE EVENT: {ACTIVE_EVENT}")

    watch = ref.on_snapshot(on_snapshot)

    stream = RTSPStream(rtsp_url)
    INVALID_FRAME_COUNT = 0

    while True:
        frame = stream.read()
        if frame is not None:
            q_detect.put(frame)
            q_record.put(frame)
            if ACTIVE_EVENT:
                q_track.put(frame)
            INVALID_FRAME_COUNT = 0
        else:
            print(f"[{cam_name}] Invalid frame received.")
            INVALID_FRAME_COUNT += 1
            time.sleep(0.1)
            if INVALID_FRAME_COUNT >= 10:
                break
        time.sleep(0.2)

    print(f"\n[{cam_name}] Too many invalid frames. Stopping stream.\n")
    stream.stop()
    watch.unsubscribe()

def threaded_process(rtsp_url, cam_id, cam_name, school,
                     detection_model, yolo, reid_model, reid_transform):
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
        args=(q_detect, cam_id, cam_name, school, detection_model),
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
