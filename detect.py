import queue
import utils as utils

def detect(frame, cam_name, infer_weapon, output_dir, grayscale=False):
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
    utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, cam_name, output_dir)
    
    # Return detection data for GUI annotation
    return boxes_np, scores_np, classes_np, valid_detections

def detect_worker(q_detect, q_display, cam_id, cam_name, school, infer_weapon, output_dir, shutdown_flag=None):
    import time
    
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
                
                # Process every frame that comes to detection queue
                detection_result = detect(frame, cam_name, infer_weapon, output_dir)
                
                # Create annotated frame for GUI display
                if q_display is not None and detection_result is not None:
                    boxes_np, scores_np, classes_np, valid_detections = detection_result
                    # Create annotated frame using existing utils function
                    annotated_frame, _ = utils.draw_bbox(frame.copy(), (boxes_np, scores_np, classes_np, valid_detections), show_label=True)
                    
                    try:
                        q_display.put((cam_id, cam_name, annotated_frame), timeout=0.05)
                    except queue.Full:
                        pass  # Skip if GUI queue is full
                elif q_display is not None:
                    # No detections, send original frame
                    try:
                        q_display.put((cam_id, cam_name, frame.copy()), timeout=0.05)
                    except queue.Full:
                        pass  # Skip if GUI queue is full
                    
            except queue.Empty:
                # No frames available - check if we should timeout
                break
            except Exception as e:
                print(f"[ERROR] Detection worker error for {cam_name}: {e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[INFO] {cam_name} detection worker received keyboard interrupt")