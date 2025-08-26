#!/usr/bin/env python3
"""
Transfer Learning approach for weapon detection using a pre-trained backbone.
This script creates a new detection model using transfer learning instead of 
trying to fine-tune the problematic TFSMLayer model.
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class COCODataLoader:
    """Load and process COCO format dataset for weapon detection"""
    
    def __init__(self, dataset_path, image_size=(224, 224)):
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

    def convert_bbox_to_normalized(self, bbox, original_size, target_size):
        """Convert COCO bbox [x, y, w, h] to normalized coordinates"""
        x, y, w, h = bbox
        orig_w, orig_h = original_size
        
        # Normalize to [0, 1] based on original image size
        x_norm = x / orig_w
        y_norm = y / orig_h
        w_norm = w / orig_w
        h_norm = h / orig_h
        
        return [x_norm, y_norm, w_norm, h_norm]

    def create_detection_dataset(self, max_boxes_per_image=5):
        """Create dataset for object detection training"""
        images = []
        targets = []
        
        for image_info in self.images.values():
            try:
                # Load and preprocess image
                image, original_size = self.load_image(image_info)
                
                # Get annotations for this image
                image_annotations = self.get_annotations_for_image(image_info['id'])
                
                # Create target array [max_boxes, 5] where 5 = [x, y, w, h, class_confidence]
                target = np.zeros((max_boxes_per_image, 5), dtype=np.float32)
                
                for i, ann in enumerate(image_annotations[:max_boxes_per_image]):
                    # Convert bbox to normalized coordinates
                    bbox_norm = self.convert_bbox_to_normalized(
                        ann['bbox'], original_size, self.image_size
                    )
                    
                    # Fill target array
                    target[i, 0:4] = bbox_norm  # x, y, w, h
                    target[i, 4] = 1.0  # Gun class confidence
                
                images.append(image)
                targets.append(target)
                
            except Exception as e:
                print(f"Error processing image {image_info['file_name']}: {e}")
                continue
        
        return np.array(images), np.array(targets)

def create_detection_model(input_shape=(224, 224, 3), max_detections=5, num_classes=1):
    """Create a detection model using EfficientNet backbone"""
    
    # Load pre-trained EfficientNet backbone
    backbone = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze backbone layers initially
    backbone.trainable = False
    
    # Build detection head
    inputs = tf.keras.Input(shape=input_shape)
    
    # Feature extraction
    features = backbone(inputs, training=False)
    
    # Global pooling
    pooled = layers.GlobalAveragePooling2D()(features)
    
    # Detection head with more layers and better regularization
    x = layers.Dense(1024, activation='relu', name='detection_dense_1')(pooled)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(512, activation='relu', name='detection_dense_2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(256, activation='relu', name='detection_dense_3')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name='detection_dense_4')(x)
    x = layers.Dropout(0.2)(x)
    
    # Output layer: [max_detections * 5] for [x, y, w, h, confidence] per detection
    detection_output = layers.Dense(max_detections * 5, activation='sigmoid', name='detections')(x)
    
    # Reshape to [max_detections, 5]
    outputs = layers.Reshape((max_detections, 5), name='detection_reshape')(detection_output)
    
    model = Model(inputs, outputs, name='weapon_detection_model')
    
    return model

@tf.keras.utils.register_keras_serializable()
def detection_loss(y_true, y_pred):
    """Custom loss function for object detection"""
    
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
    
    # Combine losses
    total_loss = 5.0 * coord_loss + 1.0 * conf_loss
    
    return tf.reduce_mean(total_loss)

def create_data_augmentation():
    """Create comprehensive data augmentation pipeline"""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.15),
        layers.RandomBrightness(0.15),
        layers.RandomTranslation(0.1, 0.1),
        # Add some noise for robustness
        layers.GaussianNoise(0.01),
    ], name="data_augmentation")

def create_callbacks(checkpoint_path, patience=25):
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
            factor=0.3,
            patience=8,
            min_lr=1e-9,
            verbose=1
        ),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: 1e-3 * (0.95 ** epoch) if epoch < 50 else 1e-5 * (0.98 ** (epoch - 50)),
            verbose=0
        )
    ]
    return callbacks

def main():
    """Main training script"""
    
    # Configuration
    DATASET_PATH = 'finetune-data/weapon-watch-dataset'
    CHECKPOINT_PATH = 'models/transfer_learning_model.keras'
    INPUT_SIZE = (224, 224, 3)  # EfficientNet standard size
    MAX_DETECTIONS = 5
    BATCH_SIZE = 4  # Smaller batch size for more training steps
    EPOCHS = 150  # Increased epochs for better training
    VALIDATION_SPLIT = 0.05  # Minimal validation split to use more data for training
    
    print("🚀 Starting Transfer Learning for Weapon Detection...")
    
    # Load dataset
    print("\n📁 Loading COCO dataset...")
    data_loader = COCODataLoader(DATASET_PATH, image_size=INPUT_SIZE[:2])
    images, targets = data_loader.create_detection_dataset(max_boxes_per_image=MAX_DETECTIONS)
    
    print(f"Dataset shape: {images.shape}, Targets shape: {targets.shape}")
    
    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, targets, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Create model
    print("\n🔧 Creating detection model...")
    model = create_detection_model(
        input_shape=INPUT_SIZE,
        max_detections=MAX_DETECTIONS,
        num_classes=1
    )
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=detection_loss,
        metrics=['mae']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Create data augmentation
    print("\n🔄 Setting up data augmentation...")
    augmentation = create_data_augmentation()
    
    # Create callbacks
    callbacks = create_callbacks(CHECKPOINT_PATH)
    
    # Create data augmentation pipeline
    print("\n🔄 Setting up data augmentation...")
    def augment_data(x, y):
        """Apply data augmentation to training data"""
        # Apply augmentation to images
        x_aug = augmentation(x, training=True)
        # Keep targets unchanged
        return x_aug, y
    
    # Create TensorFlow datasets for better performance
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    
    # Apply augmentation and batching
    train_dataset = train_dataset.map(augment_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # First phase: Train detection head only (longer phase)
    phase1_epochs = int(EPOCHS * 0.7)  # 70% of epochs for head training
    print(f"\n🏋️ Phase 1: Training detection head for {phase1_epochs} epochs (backbone frozen)...")
    history_phase1 = model.fit(
        train_dataset,
        epochs=phase1_epochs,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Second phase: Unfreeze some backbone layers for fine-tuning
    phase2_epochs = EPOCHS - phase1_epochs
    print(f"\n🔓 Phase 2: Fine-tuning with unfrozen backbone for {phase2_epochs} epochs...")
    
    # Unfreeze the top layers of the backbone
    backbone = model.get_layer('efficientnetb0')
    backbone.trainable = True
    
    # Fine-tune from this layer onwards (unfreeze top 30% of layers)
    fine_tune_at = int(len(backbone.layers) * 0.7)
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False
    
    print(f"Unfreezing {len(backbone.layers) - fine_tune_at} layers out of {len(backbone.layers)} total layers")
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-6),  # Very low LR for fine-tuning
        loss=detection_loss,
        metrics=['mae']
    )
    
    # Update callbacks for phase 2
    callbacks_phase2 = create_callbacks('models/transfer_learning_phase2.keras', patience=20)
    
    # Continue training
    history_phase2 = model.fit(
        train_dataset,
        epochs=phase2_epochs,
        validation_data=val_dataset,
        callbacks=callbacks_phase2,
        verbose=1,
        initial_epoch=phase1_epochs
    )
    
    # Save final model
    final_model_path = 'models/weapon_detection_final.keras'
    model.save(final_model_path)
    print(f"\n✅ Final model saved to {final_model_path}")
    
    # Combine histories
    combined_history = {
        'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
        'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss'],
        'mae': history_phase1.history['mae'] + history_phase2.history['mae'],
        'val_mae': history_phase1.history['val_mae'] + history_phase2.history['val_mae']
    }
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(combined_history['loss'], label='Training Loss')
    plt.plot(combined_history['val_loss'], label='Validation Loss')
    plt.axvline(x=len(history_phase1.history['loss']), color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(combined_history['mae'], label='Training MAE')
    plt.plot(combined_history['val_mae'], label='Validation MAE')
    plt.axvline(x=len(history_phase1.history['loss']), color='red', linestyle='--', label='Fine-tuning Start')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    # Learning rate plot (if available in history)
    if 'lr' in history_phase1.history:
        all_lr = history_phase1.history['lr'] + history_phase2.history['lr']
        plt.plot(all_lr, label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.yscale('log')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('transfer_learning_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n🎉 Transfer learning completed successfully!")
    print(f"Best model saved at: {CHECKPOINT_PATH}")
    print(f"Phase 2 model saved at: models/transfer_learning_phase2.keras")
    print(f"Final model saved at: {final_model_path}")
    print("\n📊 Training completed with optimized two-phase approach:")
    print(f"  Phase 1: Detection head training ({phase1_epochs} epochs, backbone frozen)")
    print(f"  Phase 2: Fine-tuning with backbone ({phase2_epochs} epochs, top layers unfrozen)")
    print(f"  Total training samples: {len(X_train)} ({len(X_train) + len(X_val)} total images)")
    print(f"  Batch size: {BATCH_SIZE}, Total epochs: {EPOCHS}")
    print(f"  Data augmentation: Enhanced with 7 augmentation techniques")
    print(f"  Architecture: EfficientNetB0 + 4-layer detection head")

def test_model():
    """Test the trained model on a sample image"""
    try:
        # Load the trained model
        model = tf.keras.models.load_model('models/weapon_detection_final.keras', 
                                         custom_objects={'detection_loss': detection_loss})
        
        # Load a test image
        test_image_path = 'testing/guns.png'
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (224, 224))
            image_normalized = image_resized.astype(np.float32) / 255.0
            image_batch = np.expand_dims(image_normalized, axis=0)
            
            # Run inference
            predictions = model.predict(image_batch)
            print(f"\n🔍 Test inference shape: {predictions.shape}")
            print(f"Predictions:\n{predictions[0]}")
            
            # Find detections with confidence > 0.5
            detections = predictions[0]
            confident_detections = detections[detections[:, 4] > 0.5]
            print(f"Found {len(confident_detections)} confident detections")
            
        else:
            print(f"Test image not found at {test_image_path}")
            
    except Exception as e:
        print(f"Error testing model: {e}")

if __name__ == '__main__':
    main()
    print("\n🧪 Testing trained model...")
    test_model()