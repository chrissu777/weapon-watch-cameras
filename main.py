import multiprocessing

from stream import RTSPStream
from process import process

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

def streamer(rtsp_url, cam_id, school):
    stream = RTSPStream(rtsp_url)
    
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    process(stream, db, cam_id, school)

if __name__ == '__main__':    
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    
    cams = db.collection('schools').document('UMD').collection('cameras').stream()
    
    processes = []
    for cam in cams:
        cam_id = cam.id
        rtsp_url = cam.to_dict()['video link']
        
        p = multiprocessing.Process(target=streamer, args=(rtsp_url, cam_id, 'UMD',))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()