import cv2
import numpy as np
import torch
import torchreid
import threading

from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

import firebase_admin
from firebase_admin import credentials, firestore

ACTIVE = False  # Global flag controlled by Firestore snapshot listener

def track_worker(q_track, cam_id, school, model, reid_model, reid_transform):
    # Firebase init
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    school_ref = db.collection('schools').document(school)

    # Listen for "Active Event" flag
    def on_snapshot(docs, changes, ts):
        global ACTIVE
        ACTIVE = docs[0].to_dict().get('Active Event', False)

    school_ref.on_snapshot(on_snapshot)

    def get_embedding(image, box):
        x1, y1, x2, y2 = map(int, box)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or x2 - x1 < 10 or y2 - y1 < 10:
            return None
        img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = reid_transform(Image.fromarray(img)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = reid_model(img).cpu().numpy().flatten()
        return feat / np.linalg.norm(feat)

    def match_embedding(embedding, embeddings, create_new):
        best_match = None
        max_sim = 0
        for entry in embeddings:
            sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
            if sim > 0.7 and sim > max_sim:
                max_sim = sim
                best_match = entry['id']
        if best_match:
            entry = next(e for e in embeddings if e['id'] == best_match)
            entry['embedding'] = (np.array(entry['embedding']) * entry['count'] + embedding) / (entry['count'] + 1)
            entry['count'] += 1
            return best_match
        elif create_new:
            if len(embeddings) > 1000:
                embeddings.pop(0)
            new_id = len(embeddings) + 1
            school_ref.update({
                "embeddings": firestore.ArrayUnion([{
                    'embedding': embedding.tolist(),
                    'id': new_id,
                    'count': 1
                }])
            })
            return new_id

    cam_ref = school_ref.collection("cameras").document(cam_id)

    try:
        while True:
            frame = q_track.get()  # blocks until a frame arrives
            if not ACTIVE:
                school_ref.update({"embeddings": [], 'detected_cam_id': ''})
                continue

            doc = school_ref.get().to_dict()
            detected_id = doc.get("detected_cam_id", "")
            embeddings = doc.get("embeddings", [])
            embeddings = [dict(e) for e in embeddings]  # ensure mutable

            cam_ref.update({"shooter_detected": False})

            if detected_id == cam_id:
                bbox = cam_ref.get().to_dict().get("bboxes", [0, 0, 0, 0])
                if sum(bbox) == 0:
                    print(f"[{cam_id}] No bounding box found.")
                    continue

                person_boxes = model(frame, verbose=False)[0].boxes
                weapon_center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                closest_person = None
                min_distance = float('inf')

                for person_box in person_boxes:
                    conf = float(person_box.conf.item())
                    cls = int(person_box.cls.item())
                    if cls == 0 and conf > 0.3:
                        x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                        person_center = [(x1 + x2) / 2, (y1 + y2) / 2]
                        distance = np.linalg.norm(np.array(weapon_center) - np.array(person_center))
                        if distance < min_distance:
                            min_distance = distance
                            closest_person = (x1, y1, x2, y2)

                if closest_person is not None:
                    embedding = get_embedding(frame, closest_person)
                    if embedding is not None:
                        shooter_id = match_embedding(embedding, embeddings, create_new=True)
                        print(f"[{cam_id}] Shooter identified: ID {shooter_id}")
                        cam_ref.update({"shooter_detected": True})
            else:
                if len(embeddings) == 0:
                    continue
                person_boxes = model(frame, verbose=False)[0].boxes
                for person_box in person_boxes:
                    conf = float(person_box.conf.item())
                    cls = int(person_box.cls.item())
                    if cls == 0 and conf > 0.3:
                        x1, y1, x2, y2 = map(int, person_box.xyxy[0].cpu().numpy())
                        embedding = get_embedding(frame, (x1, y1, x2, y2))
                        if embedding is not None:
                            shooter_id = match_embedding(embedding, embeddings, create_new=False)
                            print(f"[{cam_id}] Person {shooter_id} re-identified.")
                            cam_ref.update({"shooter_detected": True})

    except KeyboardInterrupt:
        print(f"[{cam_id}] Shutting down.")
