import cv2
import torch

import torchvision.ops as ops
import numpy as np

# helper function to convert bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        box[0], box[1], box[2], box[3] = xmin, ymin, xmax, ymax
    return bboxes

def draw_bbox(image, bboxes, info = False, show_label=True, classes=['Gun', 'Knife', 'Rifle']):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    score = 0.0  

    out_boxes, out_scores, out_classes, num_boxes = bboxes
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes:
            continue
        
        coor = out_boxes[i]
        fontScale = 0.5
        score = out_scores[i]
        
        class_ind = int(out_classes[i])
        class_name = classes[class_ind]
        if class_name == 'Rifle':
            class_name = 'Gun'
                
        if class_name not in classes:
            continue
        else:
            bbox_color = (38, 14, 194)
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 =  (int(coor[0]), int(coor[1])), (int(coor[2]), int(coor[3]))
            cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

            if info:
                print("Object found: {}, Confidence: {:.4f}, BBox Coords (xmin, ymin, xmax, ymax): {}, {}, {}, {} ".format(class_name, score, coor[0], coor[1], coor[2], coor[3]))

            if show_label:
                bbox_mess = '%s: %.2f' % (class_name, score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                cv2.rectangle(image, c1, (int(c3[0]), int(c3[1])), bbox_color, -1) #filled

                cv2.putText(image, bbox_mess, (c1[0], int(np.float32(c1[1] - 2))), cv2.FONT_HERSHEY_SIMPLEX,fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image, score

def preprocess_frame(frame, grayscale=False):
    """Preprocess frame for model inference"""
    try:
        if grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            image_data = cv2.resize(gray_frame, (608, 608)).astype(np.float32) / 255.
        else:
            image_data = cv2.resize(frame, (608, 608)).astype(np.float32) / 255.
        
        # Add batch dimension
        image_data = image_data[np.newaxis, ...]
        return image_data
    except Exception as e:
        print(f"[ERROR] Frame preprocessing failed: {e}")
        return None

def prepare_batch_data(image_data, infer_weapon):
    """Convert preprocessed image data to appropriate format for model"""
    # Convert to torch tensor (or keep as numpy for ONNX)
    if isinstance(infer_weapon.__self__ if hasattr(infer_weapon, '__self__') else None, torch.nn.Module):
        # If using PyTorch model, convert to tensor
        batch_data = torch.from_numpy(image_data).permute(0, 3, 1, 2)  # NHWC to NCHW
        if torch.cuda.is_available():
            batch_data = batch_data.cuda()
    else:
        # For ONNX model, keep as numpy
        batch_data = image_data
    
    return batch_data

def extract_predictions(pred_bbox):
    """Extract and convert predictions to numpy arrays"""
    # Extract predictions from dictionary
    value = next(iter(pred_bbox.values()))
    
    # Convert to numpy if it's a tensor
    if isinstance(value, torch.Tensor):
        value = value.cpu().numpy()
    
    # Extract boxes and confidence scores
    boxes = value[:, :, 0:4]  # [batch, num_boxes, 4]
    pred_conf = value[:, :, 4:]  # [batch, num_boxes, num_classes]
    
    return boxes, pred_conf

def apply_nms(boxes, pred_conf, score_threshold=0.25, iou_threshold=0.5, max_detections=50):
    """Apply Non-Maximum Suppression to filter predictions"""
    num_classes = pred_conf.shape[2]
    
    # Convert to tensors for NMS
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
        score_mask = class_scores > score_threshold
        if not score_mask.any():
            continue
            
        class_boxes = boxes_tensor[score_mask]
        class_scores_filtered = class_scores[score_mask]
        
        # Apply NMS
        keep_indices = ops.nms(class_boxes, class_scores_filtered, iou_threshold=iou_threshold)
        
        # Limit to max detections per class
        keep_indices = keep_indices[:max_detections]
        
        if len(keep_indices) > 0:
            final_boxes.append(class_boxes[keep_indices])
            final_scores.append(class_scores_filtered[keep_indices])
            final_classes.append(torch.full((len(keep_indices),), class_idx, dtype=torch.float32))
    
    # Combine all classes
    if final_boxes:
        all_boxes = torch.cat(final_boxes, dim=0)
        all_scores = torch.cat(final_scores, dim=0)
        all_classes = torch.cat(final_classes, dim=0)
        
        # Limit total detections
        if len(all_boxes) > max_detections:
            # Sort by score and keep top detections
            sorted_indices = torch.argsort(all_scores, descending=True)[:max_detections]
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
    
    return boxes_np, scores_np, classes_np, valid_detections

def process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, cam_name, output_dir):
    """Process valid detections and save results"""
    # Filter out class 1.0 if present (assuming this is a background/ignore class)
    if len(classes_np) > 0 and 1.0 in classes_np:
        valid_detections = 0
    
    if valid_detections > 0:
        original_h, original_w, _ = frame.shape
        bboxes = format_boxes(boxes_np[:valid_detections], original_h, original_w)
        pred_bbox = [bboxes, scores_np, classes_np, valid_detections]
        frame, score = draw_bbox(frame, pred_bbox, info=False)
        output_path = f"{output_dir}/{cam_name}_{score}.jpg"
        cv2.imwrite(output_path, frame)
        return bboxes
    else:
        return None