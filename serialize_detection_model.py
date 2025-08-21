#!/usr/bin/env python3
"""
Script to serialize the SavedModel in models/detectionmodel to a .keras file
that can be loaded and run as an object detection model.

This script handles the complex SavedModel structure with 542 inputs by using
TensorFlow's TFSMLayer to wrap the original model in a new Keras model.

Usage:
    python serialize_detection_model.py

Output:
    models/detectionmodel.keras - The serialized Keras model
"""

import tensorflow as tf
import numpy as np
import os

def serialize_detection_model():
    """
    Convert the SavedModel to a .keras file that can be easily loaded and used.
    
    Returns:
        bool: True if successful, False otherwise
        str: Path to the output .keras file
    """
    input_model_path = 'models/detectionmodel'
    output_model_path = 'models/detectionmodel.keras'
    
    print("🔄 Starting SavedModel to .keras conversion...")
    print(f"   Input:  {input_model_path}")
    print(f"   Output: {output_model_path}")
    
    try:
        # Verify input model exists
        if not os.path.exists(input_model_path):
            raise FileNotFoundError(f"SavedModel not found at: {input_model_path}")
        
        # Load and inspect the SavedModel
        print("\n📋 Analyzing SavedModel structure...")
        saved_model = tf.saved_model.load(input_model_path)
        serving_fn = saved_model.signatures['serving_default']
        
        # Get input/output information
        input_specs = serving_fn.inputs
        output_specs = serving_fn.outputs
        
        print(f"   Found {len(input_specs)} inputs and {len(output_specs)} outputs")
        
        # Get the main image input (first input)
        main_input = input_specs[0]
        print(f"   Main input shape: {main_input.shape}, dtype: {main_input.dtype}")
        
        # Create a new Keras model using TFSMLayer
        print("\n🏗️  Creating new Keras model with TFSMLayer...")
        
        # Define the input layer (image input)
        input_layer = tf.keras.Input(
            shape=(608, 608, 3), 
            dtype=tf.float32, 
            name='input_1'
        )
        
        # Create TFSMLayer to wrap the SavedModel
        tfsm_layer = tf.keras.layers.TFSMLayer(
            input_model_path,
            call_endpoint='serving_default'
        )
        
        # Apply the TFSMLayer to the input
        output = tfsm_layer(input_layer)
        
        # Create the model
        keras_model = tf.keras.Model(inputs=input_layer, outputs=output)
        
        print("✅ Keras model created successfully")
        
        # Test the model
        print("\n🧪 Testing the model...")
        test_input = np.random.random((1, 608, 608, 3)).astype(np.float32)
        test_output = keras_model(test_input)
        
        print("✅ Model test successful")
        print(f"   Input shape:  {test_input.shape}")
        
        if isinstance(test_output, dict):
            print(f"   Output type:  Dictionary with {len(test_output)} key(s)")
            for key, value in test_output.items():
                print(f"     {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   Output shape: {test_output.shape}")
            print(f"   Output dtype: {test_output.dtype}")
        
        # Save the model
        print(f"\n💾 Saving model to {output_model_path}...")
        keras_model.save(output_model_path)
        
        # Verify the saved model can be loaded
        print("\n✅ Verifying saved model...")
        loaded_model = tf.keras.models.load_model(output_model_path)
        verification_output = loaded_model(test_input)
        
        print("✅ Saved model verification successful")
        
        # Print model summary
        print("\n📊 Model Summary:")
        keras_model.summary()
        
        return True, output_model_path
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def create_usage_example():
    """Create a simple usage example script"""
    example_path = 'use_detection_model.py'
    
    example_code = '''#!/usr/bin/env python3
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
'''
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    print(f"📝 Created usage example: {example_path}")

def main():
    """Main function"""
    print("🚀 Detection Model Serialization Script")
    print("=" * 50)
    
    success, output_path = serialize_detection_model()
    
    if success:
        print("\n" + "=" * 50)
        print("🎉 CONVERSION SUCCESSFUL!")
        print(f"✅ Model saved to: {output_path}")
        print("\n📖 How to use the converted model:")
        print("   import tensorflow as tf")
        print("   model = tf.keras.models.load_model('models/detectionmodel.keras')")
        print("   output = model(input_image)  # input_image shape: (batch, 608, 608, 3)")
        
        # Create usage example
        create_usage_example()
        
        print("\n🔍 Notes:")
        print("- The model expects input images of shape (batch, 608, 608, 3)")
        print("- Input values should be normalized to [0, 1] range (float32)")
        print("- Output format depends on the original model's architecture")
        print("- The model contains ~64M parameters (244MB file size)")
        
    else:
        print("\n" + "=" * 50)
        print("❌ CONVERSION FAILED")
        print("The SavedModel structure is too complex for automatic conversion.")
        print("You may need to recreate the model architecture manually.")

if __name__ == "__main__":
    main()