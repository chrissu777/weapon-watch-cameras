# import cv2
# import io
# import numpy as np
# import tensorflow as tf
# from PIL import Image

# import utils as utils

# import firebase_admin
# from firebase_admin import credentials
# from firebase_admin import firestore
# from firebase_admin import storage

# def detect(frame, cam_id, cam_name, infer_weapon, buffer):
#     image_data = cv2.resize(frame, (608, 608))
#     image_data = image_data / 255.
#     image_data = image_data[np.newaxis, ...].astype(np.float32)

#     batch_data = tf.constant(image_data)
#     pred_bbox = infer_weapon(batch_data)

#     for key, value in pred_bbox.items():
#         boxes = value[:, :, 0:4]
#         pred_conf = value[:, :, 4:]

#     boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
#         boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
#         scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
#         max_output_size_per_class=50,
#         max_total_size=50,
#         iou_threshold=0.5,
#         score_threshold=0.35
#     )
#     valid_detections = valid_detections.numpy()[0]

#     if 1.0 in classes.numpy()[0].tolist(): valid_detections = 0

#     if valid_detections:
#         print(f"\nWEAPON DETECTED: {cam_name}")
#         # school_ref.update({"detected_cam_id": cam_id})
#         # cam_ref.update({"detected": True})
                    
#         original_h, original_w, _ = frame.shape
#         bboxes = utils.format_boxes(boxes.numpy()[0][:valid_detections], original_h, original_w)
#         # cam_ref.update({"bboxes": bboxes.flatten().tolist()})

#         pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections]
#         frame = utils.draw_bbox(frame, pred_bbox, info=False)

#         # image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         # image_pil.save(buffer, format="JPEG")
#         # buffer.seek(0)
#         # blob.upload_from_file(buffer, content_type="image/jpeg")
#         # print("DETECTED PHOTO UPLOADED TO FIREBASE")
#     # else:
#         # print(f"\nNO WEAPON DETECTED: {cam_name}")
#         # cam_ref.update({"detected": False})
#         # cam_ref.update({"bboxes": [0, 0, 0, 0]})
#         # school_ref.update({"detected_cam_id": ""})
    
#     # if frame is not None and frame.size > 0:
#     #     # cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
#     #     # cv2.imshow('Footage', frame)
#     #     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     #     #     return False
#     #     cv2.imwrite(f"detected_frames/{cam_name}.jpg", frame)
#     # else:
#     #     print("Warning: Received an empty or invalid frame")

import cv2
import numpy as np
import utils as utils
import tensorflow as tf
from firebase_admin import storage
import io
from PIL import Image
# import a utility function for loading Roboflow models
# from inference import get_model



def detect(frame, cam_id, cam_name, infer_weapon, q_display, grayscale, blob, school_ref, cam_ref, buffer):
    # if grayscale:
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
    # image_data = cv2.resize(gray_frame, (608, 608)).astype(np.float32) / 255.
    # image_data = image_data[np.newaxis, ...]

    # # model = get_model(model_id="weapon-watch-detection-model-hmqjk/1")
    # # # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    # # image = cv2.resize(gray_frame, (640, 640)).astype(np.float32) / 255.
    # # results = model.infer(image)

    # # # 1. Convert model output to a consistent tensor
    # # batch_data = tf.constant(image_data)
    # # pred_bbox = infer_weapon(batch_data)
    # # value = next(iter(pred_bbox.values()))
    
    # # # 2. Safely extract shape dimensions
    # # shape_val = tf.shape(value)
    # # batch = shape_val[0]
    # # num_classes = shape_val[-1] 
    
    # # # 3. Slice into boxes and confidences
    # # boxes = value[:, :, :4]  # shape: [batch, num_boxes, 4]
    # # pred_conf = value[:, :, 4:]  # shape: [batch, num_boxes, num_classes]
    
    # # # 4. Reshape to the required format
    # # boxes_reshaped = tf.reshape(boxes, (batch, -1, 1, 4))
    # # scores_reshaped = tf.reshape(pred_conf, (batch, -1, num_classes))
    
    # # # 5. Run combined NMS correctly
    # # boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    # #     boxes=boxes_reshaped,
    # #     scores=scores_reshaped,
    # #     max_output_size_per_class=50,
    # #     max_total_size=50,
    # #     iou_threshold=0.5,
    # #     score_threshold=0.25
    # # )

    # batch_data = tf.constant(image_data) 
    # pred_bbox = infer_weapon(batch_data) 
    # value = next(iter(pred_bbox.values())) 
    # boxes = value[:, :, 0:4] 
    # pred_conf = value[:, :, 4:] 
    
    # boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression( 
    #     boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)), 
    #     scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])), 
    #     max_output_size_per_class=50, 
    #     max_total_size=50, 
    #     iou_threshold=0.5, 
    #     score_threshold=0.25 
    #     )

    # valid_detections = valid_detections.numpy()[0]
    # boxes_np = boxes.numpy()[0]
    # scores_np = scores.numpy()[0]
    # classes_np = classes.numpy()[0]

    # if 1.0 in classes_np:
    #     valid_detections = 0

    # if valid_detections:
    #     original_h, original_w, _ = frame.shape
    #     bboxes = utils.format_boxes(boxes_np[:valid_detections], original_h, original_w)
    #     pred_bbox = [bboxes, scores_np, classes_np, valid_detections]
    #     frame, score = utils.draw_bbox(frame, pred_bbox, info=False)

    #     print(f"\nWEAPON DETECTED: {cam_name}")
    #     school_ref.update({"detected_cam_id": cam_id})
    #     cam_ref.update({"detected": True})
    #     image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    #     image_pil.save(buffer, format="JPEG")
    #     buffer.seek(0)
    #     blob.upload_from_file(buffer, content_type="image/jpeg")
        
    q_display.put((cam_name, frame))

        
def detect_worker(q_detect, cam_id, cam_name, school, infer_weapon, q_display, db):
    # if not firebase_admin._apps:
    #     cred = credentials.Certificate("serviceAccountKey.json")
    #     firebase_admin.initialize_app(cred, {
    #         "storageBucket": "weapon-watch.firebasestorage.app"
    #     })

    # db = firestore.client()
    bucket = storage.bucket()
    blob = bucket.blob(f"frame_for_verifier*{cam_id}.jpg")
    school_ref = db.collection("schools").document(school)
    cam_ref = school_ref.collection("cameras").document(cam_id)
    buffer = io.BytesIO()

    print(f"DETECTION WORKER READY FOR {cam_name}")
    
    while True:
        frame = q_detect.get()    # blocks until a frame arrives
        # detect(frame, cam_id, cam_name, detection_model, blob, school_ref, cam_ref, buffer)
        detect(frame, cam_id, cam_name, infer_weapon, q_display, True, blob, school_ref, cam_ref, buffer)