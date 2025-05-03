import multiprocessing
import os

from process import process

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

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
cam_ids = []
rtsp_urls = []
for cam in cams:
    cam_ids.append(cam.id)
    rtsp_urls.append(cam.to_dict()['video link'])

p = multiprocessing.Process(target=process, args=(rtsp_urls[1], cam_ids[1], 'UMD',))
processes.append(p)
p.start()

for p in processes:
    p.join()
"""      
    for cam in cams:
        cam_id = cam.id
        rtsp_url = cam.to_dict()['video link']
        
        p = multiprocessing.Process(target=process, args=(rtsp_url, cam_id, 'UMD',))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
"""