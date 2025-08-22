import os
import threading
import time
import signal
import shutil
import sys
import queue
import cv2

from process import threaded_process

# Global shutdown flag
shutdown_flag = threading.Event()

import torch
from ultralytics import YOLO

import onnxruntime as ort
import numpy as np
import tensorflow as tf

import firebase_admin
from firebase_admin import credentials, firestore

def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print('\n\n[INFO] Ctrl+C pressed. Shutting down ...')
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

    # Model selection and options - check command line arguments
    model_type = 'onnx'  # default
    show_frame = False  # default
    
    if len(sys.argv) > 1:
        model_arg = sys.argv[1].lower()
        if model_arg in ['onnx', 'keras', 'savedmodel']:
            model_type = model_arg
        else:
            print(f"[ERROR] Invalid model type: {model_arg}")
            print("[INFO] Usage: python main.py [onnx|keras|savedmodel] [--show-frame]")
            print("[INFO] Defaulting to ONNX model")
    
    # Check for show-frame option
    display_queue = None
    display_ready = None
    if '--show-frame' in sys.argv:
        display_queue = queue.Queue(maxsize=1)  # Small queue to force synchronization
        display_ready = threading.Event()
        display_ready.set()  # Initially ready
        print("[INFO] Frame display enabled")
    
    # Load shared detection model based on type
    print(f"\n[INFO] Loading {model_type.upper()} detection model...")
    
    if model_type == 'onnx':
        # ONNX model loading
        model_path = "models/detectionmodel.onnx"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
            
        providers = []
        if torch.cuda.is_available():
            providers.append('CUDAExecutionProvider')
        # Skip CoreML provider due to dynamic shape issues
        providers.append('CPUExecutionProvider')  # Use CPU for stability
        
        ort_session = ort.InferenceSession(model_path, providers=providers)
        
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
        
        def infer_weapon(input_tensor):
            """ONNX inference wrapper"""
            with onnx_lock:
                try:
                    if isinstance(input_tensor, torch.Tensor):
                        input_data = input_tensor.cpu().numpy()
                    elif isinstance(input_tensor, np.ndarray):
                        input_data = input_tensor
                    else:
                        input_data = np.array(input_tensor)
                    
                    if input_data.dtype != np.float32:
                        input_data = input_data.astype(np.float32)
                    
                    outputs = ort_session.run(output_names, {input_name: input_data})
                    del input_data
                    
                    if len(outputs) == 1:
                        return {output_names[0]: outputs[0]}
                    else:
                        return {name: output for name, output in zip(output_names, outputs)}
                        
                except Exception as e:
                    print(f"[ERROR] ONNX inference error: {e}")
                    return {output_names[0]: np.array([])}
    
    elif model_type == 'keras':
        # Keras model loading
        model_path = "models/detectionmodel.keras"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Keras model not found: {model_path}")
            
        keras_model = tf.keras.models.load_model(model_path)
        print(f"[INFO] Keras model loaded successfully")
        print(f"[INFO] Input shape: {keras_model.input_shape}")
        
        def infer_weapon(input_tensor):
            """Keras inference wrapper"""
            try:
                if isinstance(input_tensor, torch.Tensor):
                    input_data = input_tensor.cpu().numpy()
                elif isinstance(input_tensor, np.ndarray):
                    input_data = input_tensor
                else:
                    input_data = np.array(input_tensor)
                
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                
                outputs = keras_model(input_data)
                
                # Handle different output formats
                if isinstance(outputs, dict):
                    return outputs
                else:
                    return {'output_0': outputs}
                    
            except Exception as e:
                print(f"[ERROR] Keras inference error: {e}")
                return {'output_0': np.array([])}
    
    elif model_type == 'savedmodel':
        # SavedModel loading
        model_path = "models/detectionmodel"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SavedModel not found: {model_path}")
            
        saved_model = tf.saved_model.load(model_path)
        infer_fn = saved_model.signatures['serving_default']
        print(f"[INFO] SavedModel loaded successfully")
        print(f"[INFO] Input signature: {list(infer_fn.structured_input_signature[1].keys())}")
        
        def infer_weapon(input_tensor):
            """SavedModel inference wrapper"""
            try:
                if isinstance(input_tensor, torch.Tensor):
                    input_data = input_tensor.cpu().numpy()
                elif isinstance(input_tensor, np.ndarray):
                    input_data = input_tensor
                else:
                    input_data = np.array(input_tensor)
                
                if input_data.dtype != np.float32:
                    input_data = input_data.astype(np.float32)
                
                # Call with the main input (image)
                outputs = infer_fn(input_1=tf.constant(input_data))
                
                # Convert TensorFlow outputs to numpy and return as dict
                result = {}
                for key, value in outputs.items():
                    result[key] = value.numpy()
                
                return result
                    
            except Exception as e:
                print(f"[ERROR] SavedModel inference error: {e}")
                return {'output_0': np.array([])}
    
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
    
    # Note: Display will be handled in main thread loop below
    
    print("\n[INFO] Starting video processing...")
    if display_queue is not None:
        print("[INFO] Press 'q' in detection window or Ctrl+C to stop\n")
    else:
        print("[INFO] Press Ctrl+C to stop\n")
    
    for i in range (1):
        # rtsp_url = f'footage/cam{i}.mp4'
        # rtsp_url = f'finals_verification_vids/phase1-rifle-continuous/cam{i}_alex_2.mp4'
        rtsp_url = 'videos/rifle2.MOV'
        cam_id = i
        cam_name = f'Cam-{i}'
        
        reid_model = 0
        reid_transform = 0

        t = threading.Thread(
            target=threaded_process,
            args=(rtsp_url, cam_id, cam_name, 'UMD', infer_weapon, yolo, reid_model, reid_transform, i, output_dir, shutdown_flag, display_queue, display_ready),
            name=f"{cam_name}-main-thread",
            daemon=False  # Don't kill threads on main exit - let them finish naturally
        )
        threads.append(t)
        t.start() 
        
    try:
        # Main loop - handle display and check for thread completion
        print("[INFO] Starting display...")
        while any(t.is_alive() for t in threads) and not shutdown_flag.is_set():
            # Handle frame display in main thread
            if display_queue is not None:
                try:
                    # Wait for frames to display (blocking to sync with frame reader)
                    cam_name, frame = display_queue.get(timeout=0.1)
                    
                    # Display the frame
                    window_name = f'Detection - {cam_name}'
                    cv2.imshow(window_name, frame)
                    
                    # Signal that frame has been displayed
                    display_ready.set()
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print(f"\n[INFO] 'q' pressed in {window_name} - shutting down")
                        shutdown_flag.set()
                        break
                        
                except queue.Empty:
                    # No frame available, just update windows
                    cv2.waitKey(1)
                except Exception as e:
                    print(f"[ERROR] Display error: {e}")
            else:
                # No display, just sleep
                time.sleep(0.1)
                
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
    
    print("\n[INFO] Waiting for threads to complete, might need to Ctrl+C again")
    for t in threads:
        if t.is_alive():
            t.join(timeout=3.0)  # Wait max 3 seconds per thread
    
    # Clean up OpenCV windows
    if display_queue is not None:
        cv2.destroyAllWindows()
    
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