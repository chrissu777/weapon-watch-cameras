import numpy as np
import tensorflow as tf
import cv2
import utils as utils
import json
import time

def detect(rtsp_stream, camera_path):
    print("LOADING MODEL...\n")
    path = 'detectionmodel'
    detect_weapon = tf.saved_model.load(path)
    print("\nMODEL LOADED\n")

    start_time = time.time()
    frame_count = 1
    
    while True:
        frame = rtsp_stream.read()
        
        if frame is None:
            rtsp_stream.stop()
            break        

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
            camera_path["detected"] = True
            print("gun detected")
            
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0][:valid_detections], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections]

            frame = utils.draw_bbox(frame, pred_bbox, info=False)

        if frame is not None and frame.size > 0:
            cv2.putText(frame, "FPS: {:.3f}".format(frame_count / (time.time() - start_time)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
            cv2.imshow('Footage', frame)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
                            
            frame_count += 1
        else:
            print("Warning: Received an empty or invalid frame")

    cv2.destroyAllWindows()