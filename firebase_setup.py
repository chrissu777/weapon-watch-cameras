import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import json

with open("camera_info.json") as f:
    cams = json.load(f)['cameras']

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
cameras_col = db.collection('schools').document('UMD').collection('cameras')

for i, cam in enumerate(cams):
    update_time, camera_ref = cameras_col.add(cam)
    print(f"Added camera {i+1} with id {camera_ref.id}")

print("\nUploaded initial camera info to Firebase")