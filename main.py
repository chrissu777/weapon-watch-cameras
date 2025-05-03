import multiprocessing
import os

from process import process

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

if __name__ == '__main__':   
    # Suppress logging warnings
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"
         
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })
    db = firestore.client()
    
    cams = db.collection('schools').document('UMD').collection('cameras').stream()
    
    processes = []
    for i, cam in enumerate(cams):
        cam_id = cam.id
        cam_name = cam.to_dict()['name']
        # rtsp_url = cam.to_dict()['video link']
        video_link = f"footage/cam{cam_name[-1]}.mp4"
        
        p = multiprocessing.Process(target=process, args=(video_link, cam_id, video_link, 'UMD',))
        processes.append(p)
        p.start()
        
        if i == 2:
            break
        
    for p in processes:
        p.join()