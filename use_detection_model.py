#!/usr/bin/env python3
"""
Example script showing how to use the serialized detection model.
"""

import tensorflow as tf
import numpy as np
import cv2

def load_and_use_model():
    """Load the serialized model and run inference"""
    
    # Load the model
    print("Loading detection model...")
    model = tf.keras.models.load_model('models/detectionmodel.keras')
    print("✅ Model loaded successfully")
    
    # Prepare a test image (replace with your actual image)
    # This creates a random image for demonstration
    test_image = np.random.random((608, 608, 3)).astype(np.float32)
    
    # Add batch dimension
    input_batch = np.expand_dims(test_image, axis=0)
    print(f"Input shape: {input_batch.shape}")
    
    # Run inference
    print("Running inference...")
    detections = model(input_batch)
    
    # Process output
    if isinstance(detections, dict):
        print("Detection results (dictionary format):")
        for key, value in detections.items():
            print(f"  {key}: shape={value.shape}")
            if value.shape[1] > 0:  # Check if there are detections
                print(f"    Found {value.shape[1]} detections")
                # Process detections here based on your model's output format
            else:
                print("    No detections found")
    else:
        print(f"Detection results shape: {detections.shape}")
    
    return detections

if __name__ == "__main__":
    results = load_and_use_model()
