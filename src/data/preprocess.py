"""
Data preprocessing and augmentation for the Smart Sorting System.
Handles loading, preprocessing, and augmenting fruit and vegetable images.
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (
    MobileNetV2,
    ResNet50,
    EfficientNetB0,
    imagenet_utils
)
from tensorflow.keras.layers import Input
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles data loading, preprocessing, and augmentation."""
    
    def __init__(self, config_path='../../config/config.yaml'):
        """Initialize the data preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create necessary directories
        self._create_directories()
        
        # Initialize image size
        self.img_height, self.img_width = self.config['model']['input_shape'][:2]
        
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        paths = [
            self.config['paths']['processed_dir'],
            self.config['paths']['augmented_dir'],
            self.config['paths']['models_dir'],
            self.config['paths']['logs_dir']
        ]
        
        for path in paths:
            os.makedirs(path, exist_ok=True)
    
    def load_data(self, data_dir=None):
        """
        Load and preprocess the dataset.
        
        Args:
            data_dir: Directory containing the dataset. If None, uses config value.
            
        Returns:
            tuple: (X, y) where X is a list of image arrays and y is a list of labels
        """
        if data_dir is None:
            data_dir = self.config['paths']['data_dir']
            
        X, y = [], []
        class_names = sorted(os.listdir(data_dir))
        
        print(f"Found {len(class_names)} classes: {class_names}")
        
        # Map class names to indices
        self.class_indices = {class_name: i for i, class_name in enumerate(class_names)}
        self.num_classes = len(class_names)
        
        # Load images and labels
        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            print(f"Loading {class_name} images...")
            
            for img_name in tqdm(os.listdir(class_dir)):
                img_path = os.path.join(class_dir, img_name)
                
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image
                img = cv2.resize(img, (self.img_width, self.img_height))
                
                # Normalize pixel values to [0, 1]
                img = img.astype('float32') / 255.0
                
                X.append(img)
                y.append(self.class_indices[class_name])
        
        return np.array(X), np.array(y)
    
    def get_data_generators(self):
        """
        Create data generators for training and validation.
        
        Returns:
            tuple: (train_generator, val_generator, test_generator)
        """
        # Data augmentation configuration
        train_datagen = ImageDataGenerator(
            rotation_range=self.config['data_augmentation']['rotation_range'],
            width_shift_range=self.config['data_augmentation']['width_shift_range'],
            height_shift_range=self.config['data_augmentation']['height_shift_range'],
            shear_range=self.config['data_augmentation']['shear_range'],
            zoom_range=self.config['data_augmentation']['zoom_range'],
            horizontal_flip=self.config['data_augmentation']['horizontal_flip'],
            fill_mode=self.config['data_augmentation']['fill_mode'],
            validation_split=self.config['training']['train_val_split']
        )
        
        # Validation generator (only rescaling, no augmentation)
        val_datagen = ImageDataGenerator(
            validation_split=self.config['training']['train_val_split']
        )
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            self.config['paths']['data_dir'],
            target_size=(self.img_height, self.img_width),
            batch_size=self.config['training']['batch_size'],
            class_mode=self.config['training']['class_mode'],
            subset='training'
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.config['paths']['data_dir'],
            target_size=(self.img_height, self.img_width),
            batch_size=self.config['training']['batch_size'],
            class_mode=self.config['training']['class_mode'],
            subset='validation'
        )
        
        return train_generator, val_generator
    
    def save_processed_data(self, X, y, save_dir=None):
        """
        Save processed data to disk.
        
        Args:
            X: List of image arrays
            y: List of labels
            save_dir: Directory to save processed data. If None, uses config value.
        """
        if save_dir is None:
            save_dir = self.config['paths']['processed_dir']
            
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save as numpy arrays
        np.save(os.path.join(save_dir, 'X_processed.npy'), X)
        np.save(os.path.join(save_dir, 'y_processed.npy'), y)
        
        # Save class indices
        with open(os.path.join(save_dir, 'class_indices.json'), 'w') as f:
            json.dump(self.class_indices, f)
            
        print(f"Processed data saved to {save_dir}")
        
    def load_processed_data(self, load_dir=None):
        """
        Load processed data from disk.
        
        Args:
            load_dir: Directory containing processed data. If None, uses config value.
            
        Returns:
            tuple: (X, y) where X is an array of image arrays and y is an array of labels
        """
        if load_dir is None:
            load_dir = self.config['paths']['processed_dir']
            
        X = np.load(os.path.join(load_dir, 'X_processed.npy'))
        y = np.load(os.path.join(load_dir, 'y_processed.npy'))
        
        # Load class indices
        with open(os.path.join(load_dir, 'class_indices.json'), 'r') as f:
            self.class_indices = json.load(f)
            
        return X, y


def main():
    """Main function for testing the data preprocessing."""
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = preprocessor.load_data()
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"- Total samples: {len(X)}")
    print(f"- Number of classes: {len(np.unique(y))}")
    print(f"- Image shape: {X[0].shape}")
    
    # Get data generators
    print("\nCreating data generators...")
    train_gen, val_gen = preprocessor.get_data_generators()
    
    print("\nData generators created successfully!")
    print(f"- Training batches: {len(train_gen)}")
    print(f"- Validation batches: {len(val_gen)}")
    print(f"- Batch size: {train_gen.batch_size}")
    print(f"- Number of classes: {train_gen.num_classes}")


if __name__ == "__main__":
    import json
    main()
