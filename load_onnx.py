import torch
import threading

import onnxruntime as ort
import numpy as np

def load_onnx(model_path):
    providers = []
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        providers.append('CoreMLExecutionProvider')  # For Mac M1/M2
    providers.append('CPUExecutionProvider')  # Fallback
    ort_session = ort.InferenceSession(model_path, providers=providers)
        
    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    output_names = [output.name for output in ort_session.get_outputs()]

    if 'CUDAExecutionProvider' in ort_session.get_providers():
        print("[INFO] ✓ ONNX model will use GPU")
    else:
        print("[WARNING] ✗ ONNX model will use CPU only")

    # Create a thread lock for ONNX inference to prevent concurrent access
    onnx_lock = threading.Lock()

    # Create inference wrapper function with thread safety and error handling
    def infer_weapon(input_tensor):
        """
        Thread-safe wrapper function for ONNX inference.
        Includes error handling and memory optimization.
        Returns a dictionary to match TensorFlow model output format.
        """
        with onnx_lock:  # Ensure thread-safe inference
            try:
                # Convert input to numpy if it's a torch tensor
                if isinstance(input_tensor, torch.Tensor):
                    input_data = input_tensor.cpu().numpy()
                elif isinstance(input_tensor, np.ndarray):
                    input_data = input_tensor
                else:
                    # Handle other input types
                    input_data = np.array(input_tensor)
                
                # Ensure input has correct dtype (float32 is standard)
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                
                # Run inference with error handling
                outputs = ort_session.run(output_names, {input_name: input_data})
                
                # Clear any unnecessary references to free memory
                del input_data
                
                # Format output to match TensorFlow model's dictionary format
                # The TensorFlow model likely returned something like:
                # {'output_0': array} or {'detection_output': array}
                # Adjust the key name based on what your detect.py expects
                if len(outputs) == 1:
                    # Return as dictionary with the output name as key
                    # You may need to adjust this key based on your TF model
                    return {output_names[0]: outputs[0]}
                else:
                    # Multiple outputs - return as dictionary
                    return {name: output for name, output in zip(output_names, outputs)}
                    
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                print(f"[ERROR] ONNX Runtime error during inference: {e}")
                print("[INFO] Consider reducing batch size or image resolution")
                # Return empty dictionary to prevent crash
                return {output_names[0]: np.array([])}
            except Exception as e:
                print(f"[ERROR] Unexpected error during inference: {e}")
                return {output_names[0]: np.array([])}

    return infer_weapon