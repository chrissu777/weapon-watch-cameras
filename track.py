import numpy as np
import tensorflow as tf
import cv2
import utils as utils
import json
import time
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from PIL import Image
import torchreid
import torch
from ultralytics import YOLO

# Load ReID model (OSNet with IBN for better performance)
reid_model = torchreid.models.build_model(
    name='osnet_ibn_x1_0',
    num_classes=1000,
    loss='softmax',
    pretrained=True
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reid_model.to(device)
reid_model.eval()

# ReID image transform
reid_transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Global shooter database
shooter_db = []
shooter_id_counter = 0

def generate_new_shooter_id():
    global shooter_id_counter
    shooter_id_counter += 1
    return shooter_id_counter

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

def match_shooter(embedding):
    best_match = None
    max_sim = 0

    # Iterate through the shooter database to find the best match
    for shooter in shooter_db:
        sim = cosine_similarity([embedding], [shooter['embedding']])[0][0]
        if sim > 0.7 and sim > max_sim:
            max_sim = sim
            best_match = shooter

    if best_match:
        # Update the moving average for the matched shooter
        best_match['embedding'] = (best_match['embedding'] * best_match['count'] + embedding) / (best_match['count'] + 1)
        best_match['count'] += 1
        return best_match['shooter_id']
    else:
        # Create a new shooter entry if no match is found
        shooter_id = generate_new_shooter_id()
        shooter_db.append({
            'embedding': embedding,  # Initialize with the current embedding
            'shooter_id': shooter_id,
            'count': 1  # Start the count for the moving average
        })
        return shooter_id

def detect(notify_q, record_q, rtsp_stream):
    print("LOADING MODEL...\n")
    path = 'detectionmodel'
    detect_weapon = tf.saved_model.load(path)

    model = YOLO("yolov8n.pt")
    print("\nMODEL LOADED\n")

    start_time = time.time()
    frame_count = 1
    recording = False
    current_shooter_id = None

    while True:
        frame = rtsp_stream.read()
        
        if frame is None:
            record_q.put("finish")
            rtsp_stream.stop()
            break

        if recording:
            record_q.put(frame)
        else:
            with open('status.json') as f:
                data = json.load(f)

            if data['confirmed'] == True:
                record_q.put("start")
                recording = True

        image_data = cv2.resize(frame, (608, 608))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)

        infer_weapon = detect_weapon.signatures['serving_default']

        batch_data = tf.constant(image_data)
        pred_bbox = infer_weapon(batch_data)

        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        valid_detections = valid_detections.numpy()[0]

        if valid_detections:
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0][:valid_detections], original_h, original_w)

            # Find the person closest to the weapon
            weapon_boxes = [b for b, c in zip(bboxes, classes.numpy()[0]) if c == 1]  # Class 1 is "weapon"
            person_boxes = model(frame, verbose=False)[0] # Class 0 is "person"

            if weapon_boxes and person_boxes:
                weapon_box = weapon_boxes[0]  # Assuming one weapon
                weapon_center = [(weapon_box[0] + weapon_box[2]) / 2, (weapon_box[1] + weapon_box[3]) / 2]

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
                        shooter_id = match_shooter(embedding)
                        if shooter_id != current_shooter_id:
                            print(f"New shooter detected: ID {shooter_id}")
                            current_shooter_id = shooter_id

                        # Draw the closest person and weapon
                        x1, y1, x2, y2 = map(int, closest_person)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Shooter ID: {shooter_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            notify_q.put(bboxes)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections]

            frame = utils.draw_bbox(frame, pred_bbox, info=False)

        if frame is not None and frame.size > 0:
            cv2.putText(frame, "FPS: {:.3f}".format(frame_count / (time.time() - start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.imshow('Footage', frame)
            key = cv2.waitKey(1)
            if key == ord('f'):
                record_q.put('finish')
                break
            if key == ord('q'):
                notify_q.put("stop")
                record_q.put("stop")
                break

            frame_count += 1
        else:
            print("Warning: Received an empty or invalid frame")

    cv2.destroyAllWindows()