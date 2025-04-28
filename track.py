import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
import torchreid
import firebase_admin
from firebase_admin import firestore




def track(stream, db, cam_id, school):
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
    def match_embedding(embedding):
        try:
            best_match = None
            max_sim = 0
            for entry in embeddings:
                sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
                if sim > 0.7 and sim > max_sim:
                    max_sim = sim
                    best_match = entry['id']
            if best_match:
                entry = next(e for e in embeddings if e['id'] == best_match)
                entry['embedding'] = (entry['embedding'] * entry['count'] + embedding) / (entry['count'] + 1)
                entry['count'] += 1
                return best_match
            else:
                # no match found, create a new embedding
                print("New embedding detected, adding to database.")
                
                if len(embeddings) > 1000:
                    embeddings.pop(0)

                id = len(embeddings) + 1
                school_ref.update({
                    "embeddings": firestore.ArrayUnion([{
                        'embedding': embedding,
                        'id': id,
                        'count': 1
                    }])
                })
                
                return id
        finally:
            pass

    model = YOLO("yolov8n.pt")
    # Load ReID model (OSNet with IBN for better performance)
    reid_model = torchreid.models.build_model(
        name='osnet_ibn_x1_0',  # Better-performing ReID model
        num_classes=1000,
        loss='softmax',
        pretrained=True
    )
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    reid_model.to(device)
    reid_model.eval()

    # ReID image transform
    reid_transform = transforms.Compose([
        transforms.Resize((256, 128)),  # Standard input size for ReID models
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
    ])

    school_ref = (
                db.collection("schools")
                .document(school)
            )
    while True:
        detected_id = school_ref.get().to_dict().get("detected cam id", "")
        embeddings = school_ref.get().to_dict().get("embeddings", [])
        
        frame = stream.read()
        if frame is None:
            stream.stop()
            break  

        # reset shooter detected flag
        cam_ref.update({
                        "shooter detected": False
                    })
        if detected_id == cam_id: # weapon detected on this camera
            cam_ref = (
                db.collection("schools")
                .document(school)
                .collection("cameras")
                .document(cam_id)
            )
            
            minx, miny, maxx, maxy = cam_ref.get().to_dict().get("bboxes", [0, 0, 0, 0])
            if minx == 0 and miny == 0 and maxx == 0 and maxy == 0:
                print("No bounding box found for camera.")
                continue
            else:
                person_boxes = model(frame, verbose=False)[0] # Class 0 is "person"
                weapon_center = [(minx + maxx) / 2, (miny + maxy) / 2]

                # Find the closest person to the weapon
                closest_person = None
                min_distance = float('inf')

                for person_box in person_boxes:
                    person_center = [(person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2]
                    distance = np.linalg.norm(np.array(weapon_center) - np.array(person_center))
                    if distance < min_distance:
                        min_distance = distance
                        closest_person = person_box

                if closest_person is not None:
                    embedding = get_embedding(frame, closest_person)
                    if embedding is not None:
                        id = match_embedding(embedding)
                        print(f"New shooter detected: ID {id}")
                        cam_ref.update({
                            "shooter detected": True
                        })
        
        if len(embeddings) > 0: # check if any people match embeddings
            for entry in embeddings:
                embedding = entry['embedding']
                id = entry['id']
                sim = cosine_similarity([embedding], [entry['embedding']])[0][0]
                if sim > 0.7:
                    cam_name = cam_ref.get().to_dict().get("name", "Unknown")
                    print(f"Person {id} detected in camera {cam_name}")
                    cam_ref.update({
                        "shooter detected": True
                    })