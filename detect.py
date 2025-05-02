import cv2
import time
import io
import numpy as np
import tensorflow as tf
from PIL import Image

import utils as utils

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

def detect(frame, db, cam_id, cam_name, school, detection_model, start_time, frame_count, blob):
    image_data = cv2.resize(frame, (608, 608))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

    infer_weapon = detection_model.signatures['serving_default']

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
        print(f"WEAPON DETECTED: {cam_name}")
        
        school_ref = (
            db.collection("schools")
            .document(school)
        )
        school_ref.update({"detected cam id": cam_id})
        
        cam_ref = (
            school_ref
            .collection("cameras")
            .document(cam_id)
        )
        cam_ref.update({"detected": True})
                    
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes.numpy()[0][:valid_detections], original_h, original_w)
        
        cam_ref.update({"bboxes": bboxes.flatten().tolist()})

        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections]

        frame = utils.draw_bbox(frame, pred_bbox, info=False)
        
        image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        image_pil.save(buffer, format="JPEG")
        buffer.seek(0)

        blob.upload_from_file(buffer, content_type="image/jpeg")

    if frame is not None and frame.size > 0:
        # cv2.putText(frame, "FPS: {:.3f}".format(frame_count / (time.time() - start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
        # cv2.imshow('Footage', frame)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            return False
                        
        frame_count += 1
    else:
        print("Warning: Received an empty or invalid frame")
    
def detect_worker(q_detect, cam_id, school):    
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })

    db = firestore.client()
    bucket = storage.bucket()
    
    firebase_storage_path = "frame_for_verifier.jpg"
    blob = bucket.blob(firebase_storage_path)

    path = 'detectionmodel'
    detection_model = tf.saved_model.load(path)
    
    cam_ref = db.collection('schools').document(school).collection('cameras').document(cam_id)
    cam = cam_ref.get()
    data = cam.to_dict()
    cam_name = data['name']
    print(f"DETECTION MODEL LOADED FOR {cam_name}")
    
    while True:
        frame = q_detect.get()    # blocks until a frame arrives
        detect(frame, db, cam_id, cam_name, school, detection_model, time.time(), 1, blob)