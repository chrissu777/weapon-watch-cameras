import cv2
import os
import shutil
import numpy as np

import tensorflow as tf

from load_onnx import load_onnx
import utils as utils
import detection_utils

def test_onnx(frame, path):
    print("\n=== Testing Onnx Model ===")
    
    infer_weapon = load_onnx(path)
    print("✅ Onnx model loaded successfully")

    # Preprocess the frame first
    image_data = utils.preprocess_frame(frame)
    batch_data = utils.prepare_batch_data(image_data, infer_weapon)
    
    # Run inference with error handling
    try:
        pred_bbox = infer_weapon(batch_data)
    except Exception as e:
        print(f"[ERROR] onnx inference failed: {e}")
        return  # Skip this frame
    
    # Extract predictions
    boxes, pred_conf = utils.extract_predictions(pred_bbox)
    
    # Apply NMS to filter predictions
    boxes_np, scores_np, classes_np, valid_detections = utils.apply_nms(boxes, pred_conf)
    
    # Process and save detections
    utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, 'guns', 'onnx', 'testing/outputs/detected')

def test_keras(frame, path):
    """Test the Keras model"""
    print("\n=== Testing Keras Model ===")
    try:
        # Check if this is the transfer learning model
        is_transfer_learning = 'weapon_detection' in path
        
        if is_transfer_learning:
            # Load transfer learning model using utility function
            model = detection_utils.load_weapon_detection_model(path)
            if model is None:
                return
            print("✅ Transfer learning Keras model loaded successfully")
            
            # Preprocess for transfer learning model (224x224)
            image_resized = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            image_batch = np.expand_dims(image_resized, axis=0)
            
            # Run inference
            predictions = model.predict(image_batch, verbose=0)
            print(f"Transfer learning model output shape: {predictions.shape}")
            
            # Process predictions using utility functions
            confident_detections = detection_utils.process_transfer_learning_predictions(predictions, confidence_threshold=0.3)
            
            if len(confident_detections) > 0:
                print(f"Found {len(confident_detections)} confident detections")
                
                # Convert to pixel coordinates
                original_h, original_w = frame.shape[:2]
                boxes_np, scores_np, classes_np = detection_utils.convert_normalized_to_pixel_coords(
                    confident_detections, original_w, original_h
                )
                valid_detections = len(boxes_np)
                
                # Save detection results
                utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, 'guns', 'keras_transfer', 'testing/outputs/detected')
            else:
                print("No confident detections found")
                valid_detections = 0
        
        else:
            # Original Keras model handling
            model = tf.keras.models.load_model(path)
            print("✅ Original Keras model loaded successfully")
            
            # Preprocess the frame
            image_data = utils.preprocess_frame(frame)
            if image_data is None:
                return
            
            # For Keras model, we need to transpose to NCHW format if needed
            # Check if model expects NCHW or NHWC format
            batch_data = image_data  # Keep as NHWC for TensorFlow
            
            # Run inference
            try:
                pred_bbox = model(batch_data)
                
                # Convert to dictionary format if needed
                if not isinstance(pred_bbox, dict):
                    pred_bbox = {'output_0': pred_bbox}
                
            except Exception as e:
                print(f"[ERROR] Keras inference failed: {e}")
                return
            
            # Extract predictions
            boxes, pred_conf = utils.extract_predictions(pred_bbox)
            
            # Apply NMS to filter predictions  
            boxes_np, scores_np, classes_np, valid_detections = utils.apply_nms(boxes, pred_conf)
            
            # Process and save detections
            utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, 'guns', 'keras', 'testing/outputs/detected')
        
        print(f"✅ Keras model test completed with {valid_detections} detections")
        
    except Exception as e:
        print(f"[ERROR] Keras model test failed: {e}")

def test_tfmodel(frame, path):
    """Test the TensorFlow SavedModel"""
    print("\n=== Testing TensorFlow SavedModel ===")
    try:
        # Load TensorFlow SavedModel
        model = tf.saved_model.load(path)
        print("✅ TensorFlow SavedModel loaded successfully")
        
        # Get the inference function
        infer = model.signatures['serving_default']
        
        # Preprocess the frame
        image_data = utils.preprocess_frame(frame)
        if image_data is None:
            return
        
        # Convert to TensorFlow tensor
        batch_data = tf.constant(image_data, dtype=tf.float32)
        
        # Run inference
        try:
            pred_bbox = infer(batch_data)
            
            # Convert TensorFlow output to numpy-compatible format
            if isinstance(pred_bbox, dict):
                # Convert TensorFlow tensors to numpy
                for key, value in pred_bbox.items():
                    if hasattr(value, 'numpy'):
                        pred_bbox[key] = value.numpy()
            else:
                # Single output case
                if hasattr(pred_bbox, 'numpy'):
                    pred_bbox = {'output_0': pred_bbox.numpy()}
                else:
                    pred_bbox = {'output_0': pred_bbox}
            
        except Exception as e:
            print(f"[ERROR] TensorFlow SavedModel inference failed: {e}")
            return
        
        # Extract predictions
        boxes, pred_conf = utils.extract_predictions(pred_bbox)
        
        # Apply NMS to filter predictions
        boxes_np, scores_np, classes_np, valid_detections = utils.apply_nms(boxes, pred_conf)
        
        # Process and save detections
        utils.process_detections(boxes_np, scores_np, classes_np, valid_detections, frame, 'guns', 'tfmodel', 'testing/outputs/detected')
        print(f"✅ TensorFlow SavedModel test completed with {valid_detections} detections")
        
    except Exception as e:
        print(f"[ERROR] TensorFlow SavedModel test failed: {e}")

if __name__ == '__main__':
    # Load test image
    guns_img = cv2.imread('testing/yellow_pistol.png')
    
    if guns_img is None:
        print("[ERROR] Could not load test image 'testing/guns.png'")
        print("Please make sure the image exists or update the path")
        exit(1)
    
    print(f"Loaded test image with shape: {guns_img.shape}")
    
    output_dir = 'testing/outputs/detected'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Test all three model formats
    print("Starting model comparison tests...")
    
    test_onnx(guns_img, 'models/detectionmodel.onnx')
    test_keras(guns_img, 'models/weapon_detection_final.keras')
    test_tfmodel(guns_img, 'models/detectionmodel')
    
    print("\n🎯 All model tests completed!")