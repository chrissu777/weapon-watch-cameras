import cv2
import numpy as np
import utils as utils
import tensorflow as tf

def detect(frame, cam_name, infer_weapon, i, output_dir, grayscale=False):
    if grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    image_data = cv2.resize(frame, (608, 608)).astype(np.float32) / 255.
    image_data = image_data[np.newaxis, ...]

    batch_data = tf.constant(image_data)
    pred_bbox = infer_weapon(batch_data)

    value = next(iter(pred_bbox.values()))
    boxes = value[:, :, 0:4]
    pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.5,
        score_threshold=0.25
    )

    valid_detections = valid_detections.numpy()[0]
    boxes_np = boxes.numpy()[0]
    scores_np = scores.numpy()[0]
    classes_np = classes.numpy()[0]

    if 1.0 in classes_np:
        valid_detections = 0

    if valid_detections:
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes_np[:valid_detections], original_h, original_w)
        pred_bbox = [bboxes, scores_np, classes_np, valid_detections]
        frame, score = utils.draw_bbox(frame, pred_bbox, info=False)
        output_path = f"{output_dir}/{cam_name}_{i}_{score}.jpg"
        cv2.imwrite(output_path, frame)