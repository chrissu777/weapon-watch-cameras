import time
import io
import cv2
import queue
import utils as utils

from PIL import Image

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage

def detect(frame, cam_name, infer_weapon, output_dir, grayscale=False, use_rtdetr=False):
    if use_rtdetr:
        # RT-DETR model inference (Ultralytics format)
        try:
            results = infer_weapon(frame, verbose=False)
            
            # Extract detections from Ultralytics results
            boxes_np, scores_np, classes_np, valid_detections = utils.extract_rtdetr_predictions(results)
            
            # Process and save detections
            pred_bbox = utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, cam_name, output_dir)
            
        except Exception as e:
            print(f"[ERROR] RT-DETR inference failed for {cam_name}: {e}")
            return None
    else:
        # Original ONNX model inference
        # Preprocess frame
        image_data = utils.preprocess_frame(frame, grayscale)
        if image_data is None:
            return None  # Skip this frame if preprocessing failed
        
        # Prepare batch data for model
        batch_data = utils.prepare_batch_data(image_data, infer_weapon)
        
        # Run inference with error handling
        try:
            pred_bbox = infer_weapon(batch_data)
        except Exception as e:
            print(f"[ERROR] Inference failed for {cam_name}: {e}")
            return None  # Skip this frame
        
        # Extract predictions
        boxes, pred_conf = utils.extract_predictions(pred_bbox)
        
        # Apply NMS to filter predictions
        boxes_np, scores_np, classes_np, valid_detections = utils.apply_nms(boxes, pred_conf)
        
        # Process and save detections
        pred_bbox = utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, cam_name, output_dir)

    # Return detection data for GUI annotation
    if valid_detections:
        return boxes_np, scores_np, classes_np, valid_detections, pred_bbox
    return None

def detect_worker(q_detect, q_display, cam_id, cam_name, school, infer_weapon, output_dir, shutdown_flag=None, use_rtdetr=False):    
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })

    db = firestore.client()
    bucket = storage.bucket()
    
    firebase_storage_path = f"frame_for_verifier_{cam_id}.jpg"
    blob = bucket.blob(firebase_storage_path)

    school_ref = (
        db.collection("schools")
        .document(school)
    )

    cam_ref = (
        school_ref
        .collection("cameras")
        .document(cam_id)
    )

    buffer = io.BytesIO()

    frame_count = 0
    try:
        while True:
            # Check for shutdown signal
            if shutdown_flag and shutdown_flag.is_set():
                break
            try:
                # Use timeout to prevent indefinite blocking
                frame = q_detect.get(timeout=60.0)  # Longer timeout
                
                # Check for end-of-video sentinel
                if frame is None:
                    break
                
                frame_count += 1
                
                # Process every other frame to reduce computational load
                if frame_count % 2 == 0:
                    detection_result = detect(frame, cam_name, infer_weapon, output_dir, use_rtdetr=use_rtdetr)
                else:
                    detection_result = None  # Skip detection for this frame
                
                # Create annotated frame for GUI display
                if q_display is not None and detection_result is not None:
                    boxes_np, scores_np, classes_np, valid_detections, pred_bbox = detection_result
                    # Create annotated frame using existing utils function
                    annotated_frame, _ = utils.draw_bbox(frame.copy(), (boxes_np, scores_np, classes_np, valid_detections), show_label=True)
                    
                    # Update all neccesary firebase variables                  
                    image_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                    image_pil.save(buffer, format="JPEG")
                    buffer.seek(0)
                    
                    blob.upload_from_file(buffer, content_type="image/jpeg")

                    # Batch Firebase updates for better performance
                    batch = db.batch()
                    batch.update(cam_ref, {'detected': True, 'bboxes': pred_bbox.flatten().tolist()})
                    batch.update(school_ref, {'detected_cam_id': cam_id, 'firebase_storage_path': f"frame_for_verifier_{cam_id}.jpg"})
                    batch.commit()
                elif detection_result is None:
                    cam_ref.update({"bboxes": []})
                    # Attempt to pass frame to gui display
                    # try:
                    #     q_display.put((cam_id, cam_name, annotated_frame), timeout=0.05)
                    # except queue.Full:
                    #     pass
                
                # elif q_display is not None:
                #     try:
                #         q_display.put((cam_id, cam_name, frame.copy()), timeout=0.05)
                #     except queue.Full:
                #         pass
                    
            except queue.Empty:
                break
            except Exception as e:
                print(f"[ERROR] Detection worker error for {cam_name}: {e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[INFO] {cam_name} detection worker received keyboard interrupt")