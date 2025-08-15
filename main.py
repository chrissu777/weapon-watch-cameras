import os
import threading
import time
import signal
import shutil

from process import threaded_process

# Global shutdown flag
shutdown_flag = threading.Event()

import torch
from ultralytics import YOLO

import onnxruntime as ort
import numpy as np

import firebase_admin
from firebase_admin import credentials, firestore

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print('\n\n[INFO] Ctrl+C pressed. Shutting down gracefully...')
    shutdown_flag.set()
    # Don't call sys.exit(0) here - let main thread handle cleanup

if __name__ == '__main__':
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"\n[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Check CUDA capability
        capability = torch.cuda.get_device_capability(0)
        print(f"[INFO] CUDA Capability: {capability[0]}.{capability[1]}")
        
        if capability[0] >= 12:
            print("[WARNING] GPU has CUDA capability >= 12.0 (sm_120)")
            print("[INFO] This may cause compatibility issues with current PyTorch")
            print("[INFO] Models will attempt GPU usage but may fall back to CPU")

    # Suppress logs
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOG_minloglevel"] = "2"

    # Firebase initialization
    if not firebase_admin._apps:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred, {
            "storageBucket": "weapon-watch.firebasestorage.app"
        })
    db = firestore.client()

    # Load shared detection model
    print("\n[INFO] Loading ONNX detection model...")
    
    # Create ONNX Runtime session with appropriate providers
    providers = []
    if torch.cuda.is_available():
        providers.append('CUDAExecutionProvider')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        providers.append('CoreMLExecutionProvider')  # For Mac M1/M2
    providers.append('CPUExecutionProvider')  # Fallback
    
    # Load ONNX model
    ort_session = ort.InferenceSession("og_detectionmodel.onnx", providers=providers)
    
    # Get input and output names
    input_name = ort_session.get_inputs()[0].name
    input_shape = ort_session.get_inputs()[0].shape
    output_names = [output.name for output in ort_session.get_outputs()]

    print(f"[INFO] ONNX model loaded with input shape: {input_shape}")
    print(f"[INFO] Active providers: {ort_session.get_providers()}")
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
                
                # Validate input shape
                expected_batch = input_shape[0] if input_shape[0] != -1 else input_data.shape[0]
                if len(input_data.shape) != len(input_shape):
                    print(f"[WARNING] Input shape mismatch. Expected dims: {len(input_shape)}, got: {len(input_data.shape)}")
                
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
    
    print("[INFO] Detection model loaded and ready")

    # Load shared YOLO model
    print("\n[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")
    yolo.fuse()
    print("[INFO] YOLO model loaded.")

    # Load shared ReID model
    # print("\n[INFO] Loading ReID model...")
    # reid_model = torchreid.models.build_model(
    #     name='osnet_ibn_x1_0',
    #     num_classes=1000,
    #     loss='softmax',
    #     pretrained=True
    # )
    # device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # reid_model.to(device).eval()
    # print(f"[INFO] ReID model loaded on {device}\n")

    # reid_transform = transforms.Compose([
    #     transforms.Resize((256, 128)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    # Fetch cameras
    # cams = db.collection('schools').document('UMD').collection('cameras').stream()

    output_dir = 'testing/outputs/detected'
    
    # Remove existing output directory and create fresh one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"\n[INFO] Removed existing output directory: {output_dir}")
    
    os.makedirs(output_dir)
    print(f"[INFO] Created fresh output directory: {output_dir}")
    
    threads = []
    
    print("\n[INFO] Starting video processing...")
    print("[INFO] Press Ctrl+C to stop\n")
    
    for i in range (1,7):
        rtsp_url = f'footage/cam{i}.mp4'
        # rtsp_url = f'finals_verification_vids/phase1-pistol-continuous/cam{i}_joey.mp4'
        cam_id = i
        cam_name = f'Cam-{i}'
        
        reid_model = 0
        reid_transform = 0

        t = threading.Thread(
            target=threaded_process,
            args=(rtsp_url, cam_id, cam_name, 'UMD', infer_weapon, yolo, reid_model, reid_transform, i, output_dir, shutdown_flag),
            name=f"{cam_name}-main-thread",
            daemon=False  # Don't kill threads on main exit - let them finish naturally
        )
        threads.append(t)
        t.start() 
        
    try:
        # Wait for all threads to complete naturally, but check for Ctrl+C periodically
        while any(t.is_alive() for t in threads) and not shutdown_flag.is_set():
            time.sleep(0.1)  # Check every 100ms
    except KeyboardInterrupt:
        print("\n[INFO] Keyboard interrupt received")
        shutdown_flag.set()
    
    if not shutdown_flag.is_set():
        # All videos completed naturally
        print("\n[INFO] All videos processed successfully!")
        print("[INFO] Press Ctrl+C to exit program")
        # Wait for user to press Ctrl+C
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt received")
            shutdown_flag.set()
    
    print("\n[INFO] Waiting for threads to complete...")
    for t in threads:
        if t.is_alive():
            t.join(timeout=3.0)  # Wait max 3 seconds per thread
    
    print("[INFO] Program terminated successfully")
    
    # Clean up ONNX and PyTorch resources to prevent hanging
    try:
        if 'ort_session' in locals():
            del ort_session
        if 'yolo' in locals():
            del yolo
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    
    # Force exit to avoid hanging on C++ cleanup issues
    print("[INFO] Forcing program exit...")
    import os
    os._exit(0)