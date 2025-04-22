import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import json

with open("camera_info.json") as f:
    data = json.load(f)

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
batch = db.batch()

cameras_col = db.collection('schools').document('UMD').collection('cameras')

for building_name, floors in data.items():
    building_doc = cameras_col.document(building_name)
    for floor_name, cameras in floors.items():
        floor_col = building_doc.collection(floor_name)
        
        if cameras:
            for cam in cameras:
                cam_doc = floor_col.document(cam['name'])
                batch.set(cam_doc, cam)
        else:
            placeholder = floor_col.document('no cameras')
            batch.set(placeholder, {})


batch.commit()

print("Uploaded initial camera info to Firebase")