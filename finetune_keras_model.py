#!/usr/bin/env python3
"""
Fine-tune the Keras detection model using the weapon-watch-dataset in COCO format.
This script implements transfer learning to improve the existing model's performance
on the specific weapon detection task.
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class COCODataLoader:
    """Load and process COCO format dataset for weapon detection"""
    
    def __init__(self, dataset_path, image_size=(608, 608)):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.annotations_file = os.path.join(dataset_path, 'annotations', 'instances_default.json')
        self.images_dir = os.path.join(dataset_path, 'images', 'default')
        
        # Load COCO annotations
        with open(self.annotations_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        print(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
        print(f"Categories: {[cat['name'] for cat in self.categories.values()]}")

    def load_image(self, image_info):
        """Load and preprocess image"""
        image_path = os.path.join(self.images_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        original_h, original_w = image.shape[:2]
        image_resized = cv2.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        return image_normalized, (original_w, original_h)

    def get_annotations_for_image(self, image_id):
        """Get all annotations for a specific image"""
        return [ann for ann in self.annotations if ann['image_id'] == image_id]

    def convert_bbox_to_model_format(self, bbox, original_size, target_size):
        """Convert COCO bbox [x, y, w, h] to model format and normalize"""
        x, y, w, h = bbox
        orig_w, orig_h = original_size
        target_w, target_h = target_size
        
        # Scale to target size
        x_scaled = (x / orig_w) * target_w
        y_scaled = (y / orig_h) * target_h
        w_scaled = (w / orig_w) * target_w
        h_scaled = (h / orig_h) * target_h
        
        # Normalize to [0, 1]
        x_norm = x_scaled / target_w
        y_norm = y_scaled / target_h
        w_norm = w_scaled / target_w
        h_norm = h_scaled / target_h
        
        return [x_norm, y_norm, w_norm, h_norm]

    def create_dataset(self, max_detections=10):
        """Create dataset with images and corresponding targets"""
        images = []
        targets = []
        
        for image_info in self.images.values():
            try:
                # Load and preprocess image
                image, original_size = self.load_image(image_info)
                
                # Get annotations for this image
                image_annotations = self.get_annotations_for_image(image_info['id'])
                
                # Create target array matching wrapper model output shape [max_detections, 7]
                # where 7 = [x, y, w, h, confidence, class_prob, background_prob]
                target = np.zeros((max_detections, 7), dtype=np.float32)
                
                for i, ann in enumerate(image_annotations[:max_detections]):
                    # Convert bbox to normalized coordinates
                    bbox_norm = self.convert_bbox_to_model_format(
                        ann['bbox'], original_size, self.image_size
                    )
                    
                    # Fill target array to match model output format
                    target[i, 0:4] = bbox_norm  # x, y, w, h (normalized)
                    target[i, 4] = 1.0  # confidence (ground truth)
                    target[i, 5] = 1.0 if ann['category_id'] == 1 else 0.0  # Gun class
                    target[i, 6] = 0.0 if ann['category_id'] == 1 else 1.0  # Background class
                
                images.append(image)
                targets.append(target)
                
            except Exception as e:
                print(f"Error processing image {image_info['file_name']}: {e}")
                continue
        
        return np.array(images), np.array(targets)

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ])

def custom_detection_loss(y_true, y_pred):
    """Custom loss function for object detection"""
    
    # Extract components from predictions and targets
    # Assuming model outputs [batch, num_detections, 7]
    pred_boxes = y_pred[:, :, 0:4]  # x, y, w, h
    pred_confidence = y_pred[:, :, 4:5]  # confidence
    pred_classes = y_pred[:, :, 5:6]  # class scores
    
    true_boxes = y_true[:, :, 0:4]  # x, y, w, h
    true_confidence = y_true[:, :, 4:5]  # confidence (should be 1 for valid boxes)
    true_classes = y_true[:, :, 5:6]  # class id
    valid_boxes = y_true[:, :, 6:7]  # valid box indicator
    
    # Box coordinate loss (only for valid boxes)
    box_loss = tf.reduce_sum(
        valid_boxes * tf.square(pred_boxes - true_boxes),
        axis=[1, 2]
    )
    
    # Confidence loss
    confidence_loss = tf.reduce_sum(
        tf.square(pred_confidence - true_confidence),
        axis=[1, 2]
    )
    
    # Classification loss (only for valid boxes)
    class_loss = tf.reduce_sum(
        valid_boxes * tf.square(pred_classes - true_classes),
        axis=[1, 2]
    )
    
    # Combine losses with weights
    total_loss = 5.0 * box_loss + 1.0 * confidence_loss + 2.0 * class_loss
    
    return tf.reduce_mean(total_loss)

def create_trainable_wrapper_model(base_model, learning_rate=1e-5):
    """Create a trainable wrapper around the TFSMLayer model"""
    
    # Initialize the base model
    print("Initializing base model...")
    dummy_input = np.zeros((1, 608, 608, 3), dtype=np.float32)
    try:
        base_output = base_model.predict(dummy_input, verbose=0)
        print(f"Base model output type: {type(base_output)}")
        
        # Handle dictionary output from TFSMLayer
        if isinstance(base_output, dict):
            print(f"Model outputs dictionary with keys: {list(base_output.keys())}")
            # Get the main output tensor
            output_key = list(base_output.keys())[0]  # Take first output
            main_output = base_output[output_key]
            print(f"Main output shape: {main_output.shape}")
        else:
            main_output = base_output
            print(f"Direct output shape: {main_output.shape}")
            
    except Exception as e:
        print(f"Base model initialization error: {e}")
        return None
    
    # Make the base model non-trainable to avoid variable issues
    base_model.trainable = False
    
    # Create a custom layer to extract the dictionary output
    class DictOutputExtractor(layers.Layer):
        def __init__(self, output_key=None, **kwargs):
            super().__init__(**kwargs)
            self.output_key = output_key
            
        def call(self, inputs):
            base_output = base_model(inputs)
            if isinstance(base_output, dict):
                if self.output_key is None:
                    # Take the first output
                    key = list(base_output.keys())[0]
                else:
                    key = self.output_key
                return base_output[key]
            return base_output
        
        def get_config(self):
            config = super().get_config()
            config.update({"output_key": self.output_key})
            return config
    
    # Create a new wrapper model with trainable layers
    inputs = tf.keras.Input(shape=(608, 608, 3), name='input_images')
    
    # Extract tensor from dictionary output
    if isinstance(base_output, dict):
        output_key = list(base_output.keys())[0]
        base_predictions = DictOutputExtractor(output_key)(inputs)
    else:
        base_predictions = base_model(inputs)
    
    # Determine the shape and add appropriate adaptation layers
    output_shape = main_output.shape
    print(f"Working with output shape: {output_shape}")
    
    if len(output_shape) == 3 and output_shape[1] is not None:  # (batch, detections, features)
        print("Using detection-based adaptation layers")
        # Add some trainable layers for adaptation
        adapted = layers.Dense(16, activation='relu', name='adaptation_1')(base_predictions)
        adapted = layers.Dropout(0.1)(adapted)
        outputs = layers.Dense(7, activation='linear', name='final_output')(adapted)
    else:
        print("Using pooling-based adaptation layers")
        # Handle dynamic or unknown shapes
        # Flatten and reshape approach
        flattened = layers.Flatten()(base_predictions)
        adapted = layers.Dense(128, activation='relu', name='adaptation_1')(flattened)
        adapted = layers.Dropout(0.2)(adapted)
        adapted = layers.Dense(64, activation='relu', name='adaptation_2')(adapted)
        # Reshape to detection format (10 detections with 7 features each)
        outputs = layers.Dense(7 * 10, activation='linear')(adapted)
        outputs = layers.Reshape((10, 7), name='final_output')(outputs)
    
    # Create the new trainable model
    wrapper_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='finetuned_wrapper')
    
    # Compile the wrapper model
    optimizer = optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    wrapper_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    print("Created trainable wrapper model:")
    wrapper_model.summary()
    
    return wrapper_model

def prepare_model_for_finetuning(base_model, num_classes=1, learning_rate=1e-5):
    """Prepare the existing model for fine-tuning by creating a wrapper"""
    
    # Check if the model contains TFSMLayer
    has_tfsm_layer = any('tfsm' in layer.name.lower() for layer in base_model.layers)
    
    if has_tfsm_layer:
        print("Detected TFSMLayer - creating trainable wrapper model...")
        return create_trainable_wrapper_model(base_model, learning_rate)
    else:
        print("Standard Keras model detected - using direct fine-tuning...")
        # Standard approach for regular Keras models
        for layer in base_model.layers[:-1]:
            layer.trainable = False
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        base_model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        return base_model

def create_callbacks(checkpoint_path, patience=10):
    """Create training callbacks"""
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-8,
            verbose=1
        )
    ]
    return callbacks

def main():
    """Main fine-tuning script"""
    
    # Configuration
    DATASET_PATH = 'finetune-data/weapon-watch-dataset'
    MODEL_PATH = 'models/detectionmodel.keras'
    CHECKPOINT_PATH = 'models/finetuned_model.keras'
    BATCH_SIZE = 2  # Very small batch size for fine-tuning with limited data
    EPOCHS = 30  # Reduced epochs to prevent overfitting
    VALIDATION_SPLIT = 0.15  # Smaller validation split due to limited data
    
    print("🚀 Starting Keras model fine-tuning...")
    
    # Load dataset
    print("\n📁 Loading COCO dataset...")
    data_loader = COCODataLoader(DATASET_PATH)
    images, targets = data_loader.create_dataset()
    
    print(f"Dataset shape: {images.shape}, Targets shape: {targets.shape}")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, targets, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Load pre-trained model
    print("\n🔧 Loading pre-trained model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Original model loaded successfully")
    
    # Prepare for fine-tuning
    print("\n🎯 Preparing model for fine-tuning...")
    model = prepare_model_for_finetuning(model)
    
    # Create data augmentation
    print("\n🔄 Setting up data augmentation...")
    augmentation = create_data_augmentation()
    
    # Apply augmentation to training data
    def augment_data(x, y):
        # Only augment images, keep targets unchanged
        x_aug = augmentation(x, training=True)
        return x_aug, y
    
    # Create callbacks
    callbacks = create_callbacks(CHECKPOINT_PATH)
    
    # Training
    print("\n🏋️ Starting fine-tuning...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_model_path = 'models/finetuned_final.keras'
    model.save(final_model_path)
    print(f"\n✅ Final model saved to {final_model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("\n🎉 Fine-tuning completed successfully!")
    print(f"Best model saved at: {CHECKPOINT_PATH}")
    print(f"Final model saved at: {final_model_path}")

if __name__ == '__main__':
    main()