import numpy as np
import tensorflow as tf
import cv2
import utils as utils
import multiprocessing
import json
import time

from confirm import confirm
from record import record
from stream import RTSPStream

def detect(notify_q, record_q, rtsp_stream):
    print("LOADING MODEL...\n")
    path = 'camera/detectionmodel'
    detect_weapon = tf.saved_model.load(path)
    print("\nMODEL LOADED\n")

    start_time = time.time()
    frame_count = 1
    recording = False
    
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

if __name__ == "__main__":
    dic = {'detected': False,
           'confirmed': False}
    json_obj = json.dumps(dic, indent=4)
    with open('status.json', 'w') as f:
        f.write(json_obj)

    with open('rtsps.txt', 'r') as file:
        content = file.read()
        rtsp_urls = content.split('\n')
      
    rtsp_streams = []
    for rtsp_url in rtsp_urls:
        rtsp_streams.append(RTSPStream(rtsp_url))
        
    confirm_q = multiprocessing.Queue()
    record_q = multiprocessing.Queue()

    confirm_p = multiprocessing.Process(target=confirm, args=(confirm_q,))
    confirm_p.start()

    record_p = multiprocessing.Process(target=record, args=(record_q,))
    record_p.start()

    detect(confirm_q, record_q, rtsp_stream)
    rtsp_stream.stop()
    cv2.destroyAllWindows()
    
    confirm_q.close()
    record_q.close()

    confirm_q.join_thread()
    record_q.join_thread()

    confirm_p.join()
    record_p.join()  