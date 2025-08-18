import json
import os
import shutil
from pathlib import Path
import yaml
from ultralytics import YOLO
import torch

def coco_to_yolo_format(coco_json_path, images_dir, output_dir):
    """Convert COCO format annotations to YOLO format"""
    
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directories
    images_output = os.path.join(output_dir, 'images')
    labels_output = os.path.join(output_dir, 'labels')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)
    
    # Create image id to filename mapping
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    image_id_to_dims = {img['id']: (img['width'], img['height']) for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    converted_count = 0
    
    for image_id, filename in image_id_to_file.items():
        # Copy image file
        src_image = os.path.join(images_dir, filename)
        dst_image = os.path.join(images_output, filename)
        
        if os.path.exists(src_image):
            shutil.copy2(src_image, dst_image)
            
            # Create corresponding label file
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(labels_output, label_filename)
            
            width, height = image_id_to_dims[image_id]
            
            with open(label_path, 'w') as f:
                if image_id in annotations_by_image:
                    for ann in annotations_by_image[image_id]:
                        # Convert COCO bbox to YOLO format
                        x, y, w, h = ann['bbox']
                        
                        # Convert to YOLO format (center_x, center_y, width, height) normalized
                        center_x = (x + w/2) / width
                        center_y = (y + h/2) / height
                        norm_w = w / width
                        norm_h = h / height
                        
                        # Category ID - 1 (YOLO uses 0-based indexing)
                        class_id = ann['category_id'] - 1
                        
                        f.write(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
                # If no annotations, create empty file
                else:
                    pass  # Empty file for images with no annotations
            
            converted_count += 1
    
    print(f"Converted {converted_count} images to YOLO format")
    return converted_count

def create_dataset_yaml(dataset_dir, num_classes=1):
    """Create dataset.yaml file for YOLO training"""
    
    yaml_content = {
        'path': os.path.abspath(dataset_dir),
        'train': 'images',  # training images (relative to 'path')
        'val': 'images',    # validation images (relative to 'path') - using same for simplicity
        'test': 'images',   # test images (optional)
        'nc': num_classes,  # number of classes
        'names': ['Gun']    # class names
    }
    
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset.yaml at {yaml_path}")
    return yaml_path

def finetune_yolo_model():
    """Main finetuning function"""
    
    print("Starting weapon detection model finetuning...")
    
    # Paths
    coco_json_path = 'finetune-data/weapon-watch-dataset/annotations/instances_default.json'
    images_dir = 'finetune-data/weapon-watch-dataset/images/default'
    yolo_dataset_dir = 'finetune-data/yolo-dataset'
    
    # Convert COCO to YOLO format
    print("Converting COCO format to YOLO format...")
    converted_count = coco_to_yolo_format(coco_json_path, images_dir, yolo_dataset_dir)
    
    if converted_count == 0:
        print("ERROR: No images were converted. Check your dataset paths.")
        return
    
    # Create dataset.yaml
    print("Creating dataset configuration...")
    dataset_yaml = create_dataset_yaml(yolo_dataset_dir)
    
    # Load a YOLOv8 model (you can change this to different model sizes)
    print("Loading YOLOv8 model...")
    
    # Try to use the existing yolov8n.pt if available, otherwise download
    if os.path.exists('yolov8n.pt'):
        model = YOLO('yolov8n.pt')
        print("Loaded existing yolov8n.pt")
    else:
        model = YOLO('yolov8n.pt')  # This will download if not present
        print("Downloaded yolov8n.pt")
    
    # Training parameters
    train_params = {
        'data': dataset_yaml,
        'epochs': 100,           # Number of training epochs
        'imgsz': 608,           # Image size (matching your detection code)
        'batch': 16,            # Batch size (adjust based on GPU memory)
        'device': 0 if torch.cuda.is_available() else 'cpu',  # Use GPU if available
        'workers': 4,           # Number of worker threads
        'patience': 50,         # Early stopping patience
        'save': True,           # Save checkpoints
        'save_period': 10,      # Save checkpoint every N epochs
        'cache': False,         # Cache images for faster training
        'optimizer': 'SGD',     # Optimizer (SGD, Adam, AdamW)
        'lr0': 0.01,           # Initial learning rate
        'lrf': 0.01,           # Final learning rate factor
        'momentum': 0.937,      # Momentum
        'weight_decay': 0.0005, # Weight decay
        'warmup_epochs': 3,     # Warmup epochs
        'warmup_momentum': 0.8, # Warmup momentum
        'warmup_bias_lr': 0.1,  # Warmup bias learning rate
        'box': 7.5,            # Box loss gain
        'cls': 0.5,            # Class loss gain
        'dfl': 1.5,            # DFL loss gain
        'pose': 12.0,          # Pose loss gain (not used for detection)
        'kobj': 2.0,           # Keypoint object loss gain (not used for detection)
        'label_smoothing': 0.0, # Label smoothing
        'nbs': 64,             # Nominal batch size
        'overlap_mask': True,   # Overlap masks
        'mask_ratio': 4,        # Mask downsample ratio
        'dropout': 0.0,         # Dropout rate
        'val': True,           # Validate during training
        'plots': True,         # Save training plots
        'verbose': True        # Verbose output
    }
    
    print("Starting training...")
    print(f"Training on {converted_count} images")
    print(f"Using device: {train_params['device']}")
    
    # Train the model
    try:
        results = model.train(**train_params)
        print("Training completed successfully!")
        
        # Get the best model path
        best_model_path = model.trainer.best
        print(f"Best model saved at: {best_model_path}")
        
        # Load the best model for export
        best_model = YOLO(best_model_path)
        
        # Export to ONNX format
        print("Exporting model to ONNX format...")
        onnx_path = best_model.export(
            format='onnx',
            imgsz=608,
            dynamic=False,
            simplify=True,
            opset=11
        )
        
        # Copy the exported model to the desired filename
        final_model_path = 'new_detectionmodel.onnx'
        if os.path.exists(onnx_path):
            shutil.copy2(onnx_path, final_model_path)
            print(f"Model exported successfully as: {final_model_path}")
        else:
            print(f"ERROR: Could not find exported ONNX model at {onnx_path}")
        
        # Print training summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Dataset: {converted_count} images")
        print(f"Epochs completed: {len(results.results) if hasattr(results, 'results') else 'N/A'}")
        print(f"Best model: {best_model_path}")
        print(f"Final model: {final_model_path}")
        print("="*50)
        
        return final_model_path
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        return None

def validate_model():
    """Validate the trained model"""
    model_path = 'new_detectionmodel.onnx'
    
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found for validation")
        return
    
    print(f"Model {model_path} created successfully!")
    print(f"Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    # Check if dataset exists
    if not os.path.exists('finetune-data/weapon-watch-dataset/annotations/instances_default.json'):
        print("ERROR: Dataset not found. Make sure finetune-data/weapon-watch-dataset/ exists with annotations and images.")
        exit(1)
    
    if not os.path.exists('finetune-data/weapon-watch-dataset/images/default'):
        print("ERROR: Images directory not found. Make sure finetune-data/weapon-watch-dataset/images/default/ exists.")
        exit(1)
    
    # Run finetuning
    final_model = finetune_yolo_model()
    
    if final_model:
        validate_model()
        print("\nFinetuning completed successfully!")
        print(f"Your new model is ready: {final_model}")
        print("\nYou can now use this model in your detection pipeline by replacing og_detectionmodel.onnx")
    else:
        print("\nFinetuning failed. Please check the error messages above.")