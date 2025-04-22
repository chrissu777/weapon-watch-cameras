import json
import multiprocessing

from stream import RTSPStream
from process import process

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def streamer(rtsp_url, db, school, building, floor, cam_id):
    stream = RTSPStream(rtsp_url)
    process(stream, db, school, building, floor, cam_id)

if __name__ == '__main__':
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    with open('camera_info.json') as f:
        data = json.load(f)

    processes = []
    for cam in data['IDEA factory']['floor 1']:
        rtsp_url = cam['video link']
        p = multiprocessing.Process(target=streamer, args=(rtsp_url, db, 'UMD', 'IDEA factory', 'floor 1', cam['name'],))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()