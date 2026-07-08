import os
import shutil
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

def prepare_fashion_mnist(base_dir='dataset'):
    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Combine everything
    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)
    
    # Fashion MNIST class names
    classes = ['T-shirt_top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_boot']
    
    # Split into 70% train, 30% temp
    x_train, x_temp, y_train, y_temp = train_test_split(x_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    
    # Split temp into 50% val, 50% test (which means 15% each of the original total)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # Create directories and save images
    splits = {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }
    
    for split_name, (x_data, y_data) in splits.items():
        split_dir = os.path.join(base_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name in classes:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
            
        print(f"Saving {split_name} data...")
        for i, (img, label_idx) in enumerate(zip(x_data, y_data)):
            # label_idx is an integer in Fashion MNIST
            class_name = classes[label_idx]
            
            # Grayscale saving
            img_path = os.path.join(split_dir, class_name, f"{i}.png")
            cv2.imwrite(img_path, img)
            
            if i % 5000 == 0 and i > 0:
                print(f"  Saved {i}/{len(x_data)}")
        print(f"  Finished {split_name} (Total: {len(x_data)})")

if __name__ == "__main__":
    if os.path.exists('dataset'):
        shutil.rmtree('dataset')
    prepare_fashion_mnist()
