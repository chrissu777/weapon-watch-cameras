import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

import json

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)          # No databaseURL needed

db = firestore.client()

with open("camera_info.json") as f:
    data = json.load(f)

batch = db.batch()
for cam_id, cam_data in data.items():     # cam_id could be “cam‑001”, etc.
    doc_ref = (
        db.collection("schools")
          .document("UMD")
          .collection("cameras")
          .document(cam_id)
    )
    batch.set(doc_ref, cam_data)

batch.commit()
print("Uploaded initial camera info to Firebase")