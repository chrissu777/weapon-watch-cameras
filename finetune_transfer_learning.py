#!/usr/bin/env python3
"""
Transfer learning approach for finetuning the detection model.
Since the existing model uses TFSMLayer, we'll create a new detection head.
"""

import tensorflow as tf
import numpy as np
import json
import os
import cv2
from pathlib import Path
import yaml

def load_yolo_data_for_training():
    """Load and prepare YOLO format data for training"""
    train_images_dir = "finetune-data/yolo8-dataset/images/train"
    train_labels_dir = "finetune-data/yolo8-dataset/labels/train"
    val_images_dir = "finetune-data/yolo8-dataset/images/val" 
    val_labels_dir = "finetune-data/yolo8-dataset/labels/val"
    
    def load_dataset(images_dir, labels_dir):
        dataset = []
        image_files = list(Path(images_dir).glob('*.jpg'))
        
        for image_path in image_files:
            label_path = Path(labels_dir) / f"{image_path.stem}.txt"
            
            if not label_path.exists():
                continue
                
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            # Resize and normalize
            image_resized = cv2.resize(image, (608, 608))
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Parse labels
            boxes = []
            if label_path.stat().st_size > 0:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            boxes.append([x_center, y_center, width, height, class_id])
            
            # Create simple classification target (has weapon: 1, no weapon: 0)
            has_weapon = 1.0 if len(boxes) > 0 else 0.0
            
            dataset.append({
                'image': image_normalized,
                'has_weapon': has_weapon,
                'boxes': boxes
            })
        
        return dataset
    
    train_data = load_dataset(train_images_dir, train_labels_dir)
    val_data = load_dataset(val_images_dir, val_labels_dir)
    
    print(f"✅ Loaded {len(train_data)} training samples")
    print(f"✅ Loaded {len(val_data)} validation samples")
    
    return train_data, val_data

def create_simple_classifier_head():
    """Create a simple weapon classification model for finetuning"""
    print("🏗️  Creating weapon classifier for finetuning...")
    
    # Create a simple CNN for weapon detection
    input_layer = tf.keras.Input(shape=(608, 608, 3), name='input_1')
    
    # Feature extraction layers
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Classification head
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output: weapon classification
    weapon_output = tf.keras.layers.Dense(1, activation='sigmoid', name='weapon_detection')(x)
    
    model = tf.keras.Model(inputs=input_layer, outputs=weapon_output)
    
    print("✅ Simple classifier created")
    return model

def create_feature_extractor_model():
    """Create a model that uses the existing TFSMLayer as feature extractor"""
    print("🔧 Creating feature extractor model with TFSMLayer...")
    
    try:
        # Load the base model
        base_model = tf.keras.models.load_model("models/detectionmodel.keras")
        
        # Get the TFSMLayer
        tfsm_layer = None
        for layer in base_model.layers:
            if 'tfsm' in layer.name.lower():
                tfsm_layer = layer
                break
        
        if tfsm_layer is None:
            print("❌ TFSMLayer not found")
            return None
        
        # Create new model using TFSMLayer as feature extractor
        input_layer = tf.keras.Input(shape=(608, 608, 3), name='input_1')
        
        # Freeze the TFSMLayer
        tfsm_layer.trainable = False
        
        # Get features from TFSMLayer
        features = tfsm_layer(input_layer)
        
        # Handle dictionary output
        if isinstance(features, dict):
            # Use the first (and likely only) output
            feature_tensor = list(features.values())[0]
        else:
            feature_tensor = features
        
        # Add global pooling to handle variable-size output
        x = tf.keras.layers.GlobalAveragePooling1D()(feature_tensor)
        
        # Add new trainable classification head
        x = tf.keras.layers.Dense(256, activation='relu', name='classifier_1')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(128, activation='relu', name='classifier_2')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        # Final weapon detection output
        weapon_output = tf.keras.layers.Dense(1, activation='sigmoid', name='weapon_classification')(x)
        
        # Create the transfer learning model
        transfer_model = tf.keras.Model(inputs=input_layer, outputs=weapon_output)
        
        print("✅ Transfer learning model created")
        return transfer_model
        
    except Exception as e:
        print(f"❌ Error creating feature extractor model: {e}")
        import traceback
        traceback.print_exc()
        return None

def train_weapon_classifier():
    """Train a weapon classifier using the labeled data"""
    print("🎯 Training Weapon Classifier")
    print("=" * 50)
    
    # Load data
    train_data, val_data = load_yolo_data_for_training()
    
    if len(train_data) == 0:
        print("❌ No training data found")
        return False
    
    # Prepare training arrays
    X_train = np.array([sample['image'] for sample in train_data])
    y_train = np.array([sample['has_weapon'] for sample in train_data])
    
    X_val = np.array([sample['image'] for sample in val_data]) if val_data else None
    y_val = np.array([sample['has_weapon'] for sample in val_data]) if val_data else None
    
    print(f"📊 Training data: {X_train.shape}, labels: {y_train.shape}")
    print(f"📊 Positive samples: {np.sum(y_train):.0f}/{len(y_train)} ({np.mean(y_train)*100:.1f}%)")
    
    if X_val is not None:
        print(f"📊 Validation data: {X_val.shape}, labels: {y_val.shape}")
        print(f"📊 Val positive samples: {np.sum(y_val):.0f}/{len(y_val)} ({np.mean(y_val)*100:.1f}%)")
    
    # Try transfer learning model first, fallback to simple classifier
    model = create_feature_extractor_model()
    if model is None:
        print("⚠️  TFSMLayer approach failed, using simple classifier...")
        model = create_simple_classifier_head()
    
    if model is None:
        print("❌ Failed to create model")
        return False
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    print("\n📋 Model Summary:")
    model.summary()
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'models/weapon_classifier_best.keras',
            save_best_only=True,
            monitor='val_accuracy' if X_val is not None else 'accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=15,
            monitor='val_loss' if X_val is not None else 'loss',
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=7,
            monitor='val_loss' if X_val is not None else 'loss',
            verbose=1,
            min_lr=1e-7
        )
    ]
    
    # Train the model
    print("\n🚀 Starting training...")
    
    try:
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=100,
            batch_size=8,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        # Save final model
        final_model_path = "models/weapon_classifier_final.keras"
        model.save(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")
        
        # Test the model
        print("\n🧪 Testing trained model...")
        test_accuracy = model.evaluate(X_val if X_val is not None else X_train, 
                                     y_val if y_val is not None else y_train, verbose=0)
        print(f"✅ Test accuracy: {test_accuracy[1]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trained_classifier():
    """Test the trained classifier on sample images"""
    model_path = "models/weapon_classifier_best.keras"
    
    if not os.path.exists(model_path):
        print(f"❌ Trained model not found: {model_path}")
        return
    
    print(f"\n🧪 Testing classifier: {model_path}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        
        # Test on a few sample images
        test_images = [
            "finetune-data/yolo8-dataset/images/train/cam3_alex_2_frame_300.jpg",
            "finetune-data/yolo8-dataset/images/val/cam3_frame_19527.jpg"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                image_resized = cv2.resize(image, (608, 608))
                image_normalized = image_resized.astype(np.float32) / 255.0
                input_batch = np.expand_dims(image_normalized, axis=0)
                
                prediction = model(input_batch)
                confidence = float(prediction[0, 0])
                
                print(f"📷 {os.path.basename(image_path)}: {confidence:.3f} ({'WEAPON' if confidence > 0.5 else 'NO WEAPON'})")
        
    except Exception as e:
        print(f"❌ Error testing classifier: {e}")

if __name__ == "__main__":
    success = train_weapon_classifier()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 WEAPON CLASSIFIER TRAINING SUCCESSFUL!")
        print("📁 Models created:")
        print("   - models/weapon_classifier_best.keras (best validation model)")
        print("   - models/weapon_classifier_final.keras (final model)")
        
        test_trained_classifier()
        
        print("\n💡 Next Steps:")
        print("1. This creates a weapon/no-weapon classifier")
        print("2. To use with your detection system, you can:")
        print("   - Use it as a secondary filter after main detection")
        print("   - Or replace the detection model entirely")
        print("3. The model outputs a confidence score [0-1] for weapon presence")
        
    else:
        print("\n❌ TRAINING FAILED")
        print("Check error messages above for details")