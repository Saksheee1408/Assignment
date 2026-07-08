import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_custom_cnn_a, build_custom_cnn_b

def train_model(model_type='A', epochs=30, batch_size=64):
    train_dir = 'dataset/train'
    val_dir = 'dataset/val'
    
    # Preprocessing with data augmentation for training
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1]
    )
    
    # Only rescaling for validation
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Build model
    if model_type.upper() == 'A':
        model = build_custom_cnn_a(num_classes=10)
    else:
        model = build_custom_cnn_b(num_classes=10)
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    os.makedirs('models', exist_ok=True)
    model_path = f'models/best_model_{model_type.upper()}.h5'
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1)
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop]
    )
    
    # Save history as well if needed
    import pandas as pd
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(f'models/history_{model_type.upper()}.csv', index=False)
    print(f"Training complete. Model saved to {model_path}")
    
    return history

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='A', choices=['A', 'B'], help='Model architecture to use (A or B)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    args = parser.parse_args()
    
    train_model(model_type=args.model, epochs=args.epochs, batch_size=args.batch_size)
