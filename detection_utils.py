#!/usr/bin/env python3
"""
Utility functions for weapon detection models, including custom loss functions
and model loading helpers.
"""

import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
def detection_loss(y_true, y_pred):
    """
    Custom loss function for object detection.
    
    Args:
        y_true: Ground truth tensor [batch, max_detections, 5] where 5 = [x, y, w, h, confidence]
        y_pred: Predicted tensor [batch, max_detections, 5] where 5 = [x, y, w, h, confidence]
    
    Returns:
        Combined loss value
    """
    
    # Separate coordinates and confidence
    true_coords = y_true[:, :, :4]  # [batch, max_boxes, 4]
    true_conf = y_true[:, :, 4:5]   # [batch, max_boxes, 1]
    
    pred_coords = y_pred[:, :, :4]  # [batch, max_boxes, 4]
    pred_conf = y_pred[:, :, 4:5]   # [batch, max_boxes, 1]
    
    # Coordinate loss (only for boxes with objects)
    has_object = tf.cast(true_conf > 0.5, tf.float32)
    coord_loss = tf.reduce_sum(
        has_object * tf.square(pred_coords - true_coords),
        axis=[1, 2]
    )
    
    # Confidence loss
    conf_loss = tf.reduce_sum(
        tf.square(pred_conf - true_conf),
        axis=[1, 2]
    )
    
    # Combine losses with weights
    total_loss = 5.0 * coord_loss + 1.0 * conf_loss
    
    return tf.reduce_mean(total_loss)

def load_weapon_detection_model(model_path):
    """
    Load a weapon detection model with proper custom objects.
    
    Args:
        model_path: Path to the saved Keras model
        
    Returns:
        Loaded Keras model
    """
    custom_objects = {
        'detection_loss': detection_loss
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model from {model_path}: {e}")
        return None

def process_transfer_learning_predictions(predictions, confidence_threshold=0.3):
    """
    Process predictions from transfer learning weapon detection model.
    
    Args:
        predictions: Model predictions [batch, max_detections, 5]
        confidence_threshold: Minimum confidence to consider a detection
        
    Returns:
        List of detections with [x_norm, y_norm, w_norm, h_norm, confidence]
    """
    if len(predictions.shape) == 3:
        # Remove batch dimension if present
        detections = predictions[0]
    else:
        detections = predictions
    
    # Filter by confidence threshold
    confident_detections = detections[detections[:, 4] > confidence_threshold]
    
    return confident_detections

def convert_normalized_to_pixel_coords(detections, image_width, image_height):
    """
    Convert normalized bounding box coordinates to pixel coordinates.
    
    Args:
        detections: Array of detections [n, 5] where 5 = [x_norm, y_norm, w_norm, h_norm, conf]
        image_width: Original image width in pixels
        image_height: Original image height in pixels
        
    Returns:
        boxes_pixel: Array of pixel coordinates [n, 4] as [xmin, ymin, xmax, ymax]
        scores: Array of confidence scores [n]
        classes: Array of class ids [n] (all 0 for gun class)
    """
    boxes_pixel = []
    scores = []
    classes = []
    
    for det in detections:
        x_norm, y_norm, w_norm, h_norm, conf = det
        
        # Convert normalized coords to pixel coords
        x = int(x_norm * image_width)
        y = int(y_norm * image_height)
        w = int(w_norm * image_width)
        h = int(h_norm * image_height)
        
        # Convert to [xmin, ymin, xmax, ymax] format
        xmin = max(0, x)
        ymin = max(0, y)
        xmax = min(image_width, x + w)
        ymax = min(image_height, y + h)
        
        boxes_pixel.append([xmin, ymin, xmax, ymax])
        scores.append(float(conf))
        classes.append(0)  # Gun class = 0
    
    return np.array(boxes_pixel), np.array(scores), np.array(classes)

# Import numpy for the conversion function
import numpy as np