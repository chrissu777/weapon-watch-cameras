import cv2
import numpy as np
import utils as utils
import torch
import torchvision.ops as ops
import queue

def detect(frame, cam_name, infer_weapon, i, output_dir, grayscale=False, display_queue=None):
    detection_found = False
    try:
        if grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            image_data = cv2.resize(gray_frame, (608, 608)).astype(np.float32) / 255.
        else:
            image_data = cv2.resize(frame, (608, 608)).astype(np.float32) / 255.
        
        # Add batch dimension
        image_data = image_data[np.newaxis, ...]
        
        # Convert to torch tensor (or keep as numpy for ONNX)
        if isinstance(infer_weapon.__self__ if hasattr(infer_weapon, '__self__') else None, torch.nn.Module):
            # If using PyTorch model, convert to tensor
            batch_data = torch.from_numpy(image_data).permute(0, 3, 1, 2)  # NHWC to NCHW
            if torch.cuda.is_available():
                batch_data = batch_data.cuda()
        else:
            # For ONNX model, keep as numpy
            batch_data = image_data
        
        # Run inference with error handling
        try:
            pred_bbox = infer_weapon(batch_data)
        except Exception as e:
            print(f"[ERROR] Inference failed for {cam_name}: {e}")
            return  # Skip this frame
    except Exception as e:
        print(f"[ERROR] Frame preprocessing failed for {cam_name}: {e}")
        return  # Skip this frame
    
    # Extract predictions from dictionary
    value = next(iter(pred_bbox.values()))
    
    # Convert to numpy if it's a tensor (handle both PyTorch and TensorFlow tensors)
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    elif hasattr(value, 'numpy'):  # TensorFlow tensor
        value = value.numpy()
    
    # Handle different output shapes from different model types
    try:
        if len(value.shape) == 3:
            # Standard 3D output: [batch, num_boxes, features]
            boxes = value[:, :, 0:4]  # [batch, num_boxes, 4]
            pred_conf = value[:, :, 4:]  # [batch, num_boxes, num_classes]
        elif len(value.shape) == 2:
            # 2D output: [num_boxes, features] - add batch dimension
            value = value[np.newaxis, ...]  # Add batch dimension
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
        elif len(value.shape) == 1:
            # 1D output - likely empty or malformed
            print(f"[WARNING] Unexpected 1D output shape: {value.shape}")
            return False
        else:
            print(f"[WARNING] Unexpected output shape: {value.shape}")
            return False
    except Exception as e:
        print(f"[ERROR] Error parsing model output: {e}")
        print(f"[DEBUG] Output shape: {value.shape}, type: {type(value)}")
        return False
    
    # Reshape for PyTorch NMS
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1]
    num_classes = pred_conf.shape[2]
    
    # Convert to tensors for NMS (ensure numpy arrays first)
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if not isinstance(pred_conf, np.ndarray):
        pred_conf = np.array(pred_conf)
        
    boxes_tensor = torch.from_numpy(boxes.reshape(-1, 4))  # [total_boxes, 4]
    scores_tensor = torch.from_numpy(pred_conf.reshape(-1, num_classes))  # [total_boxes, num_classes]
    
    # Move to CPU to prevent GPU memory issues
    if boxes_tensor.is_cuda:
        boxes_tensor = boxes_tensor.cpu()
    if scores_tensor.is_cuda:
        scores_tensor = scores_tensor.cpu()
    
    # Apply NMS for each class
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for class_idx in range(num_classes):
        class_scores = scores_tensor[:, class_idx]
        
        # Filter by score threshold
        score_mask = class_scores > 0.4
        if not score_mask.any():
            continue
            
        class_boxes = boxes_tensor[score_mask]
        class_scores_filtered = class_scores[score_mask]
        
        # Apply NMS
        keep_indices = ops.nms(class_boxes, class_scores_filtered, iou_threshold=0.5)
        
        # Limit to max 50 detections per class
        keep_indices = keep_indices[:50]
        
        if len(keep_indices) > 0:
            final_boxes.append(class_boxes[keep_indices])
            final_scores.append(class_scores_filtered[keep_indices])
            final_classes.append(torch.full((len(keep_indices),), class_idx, dtype=torch.float32))
    
    # Combine all classes
    if final_boxes:
        all_boxes = torch.cat(final_boxes, dim=0)
        all_scores = torch.cat(final_scores, dim=0)
        all_classes = torch.cat(final_classes, dim=0)
        
        # Limit total detections to 50
        if len(all_boxes) > 50:
            # Sort by score and keep top 50
            sorted_indices = torch.argsort(all_scores, descending=True)[:50]
            all_boxes = all_boxes[sorted_indices]
            all_scores = all_scores[sorted_indices]
            all_classes = all_classes[sorted_indices]
        
        # Convert back to numpy
        boxes_np = all_boxes.numpy()
        scores_np = all_scores.numpy()
        classes_np = all_classes.numpy()
        valid_detections = len(boxes_np)
        
        # Clean up GPU memory
        del all_boxes, all_scores, all_classes
    else:
        # No detections
        boxes_np = np.array([]).reshape(0, 4)
        scores_np = np.array([])
        classes_np = np.array([])
        valid_detections = 0
    
    # Clean up tensors
    del boxes_tensor, scores_tensor
    if final_boxes:
        del final_boxes, final_scores, final_classes
    
    # Filter out class 1.0 if present (assuming this is a background/ignore class)
    if len(classes_np) > 0 and 1.0 in classes_np:
        valid_detections = 0
    
    if valid_detections > 0:
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes(boxes_np[:valid_detections], original_h, original_w)
        pred_bbox = [bboxes, scores_np, classes_np, valid_detections]
        frame, score = utils.draw_bbox(frame, pred_bbox, info=False)
        output_path = f"{output_dir}/{cam_name}_{i}_{score}.jpg"
        cv2.imwrite(output_path, frame)
        detection_found = True
    
    # Send frame to display queue if requested
    if display_queue is not None:
        try:
            # Send frame with camera name for display
            display_queue.put((cam_name, frame.copy()), timeout=0.01)
        except queue.Full:
            print("[WARNING] display queue full")
            pass  # Skip frame if display queue is full
    
    # Return whether detection was found
    return detection_found


def detect_worker(q_detect, cam_id, cam_name, school, infer_weapon, i, output_dir, shutdown_flag=None, display_queue=None):
    import time
    
    # print(f"DETECTION WORKER READY FOR {cam_name}")
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            # Check for shutdown signal
            if shutdown_flag and shutdown_flag.is_set():
                # print(f"\n[INFO] {cam_name} detection worker shutting down...")
                break
            try:
                # Use timeout to prevent indefinite blocking
                frame = q_detect.get(timeout=5.0)  # Longer timeout
                
                # Check for end-of-video sentinel
                if frame is None:
                    # print(f"[INFO] {cam_name} detection worker received end signal")
                    print(f"[INFO] {cam_name} completed with {detection_count} weapon detections")
                    break
                
                frame_count += 1
                
                # Process every frame that comes to detection queue
                had_detection = detect(frame, cam_name, infer_weapon, i, output_dir, True, display_queue)
                if had_detection:
                    detection_count += 1
                    
            except queue.Empty:
                # No frames available - check if we should timeout
                print(f"[INFO] {cam_name} detection worker timed out waiting for frames - exiting")
                print(f"[INFO] {cam_name} completed with {detection_count} weapon detections")
                break
            except Exception as e:
                print(f"[ERROR] Detection worker error for {cam_name}: {e}")
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[INFO] {cam_name} detection worker received keyboard interrupt")
        print(f"[INFO] {cam_name} completed with {detection_count} weapon detections")