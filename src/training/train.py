"""
Training script for the Smart Sorting System.
Handles the training pipeline including data loading, model training, and evaluation.
"""

import os
import sys
import yaml
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
import argparse

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.preprocess import DataPreprocessor
from src.models.model_builder import ModelBuilder

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model for fruit/vegetable classification.')
    parser.add_argument('--config', type=str, default='../../config/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model', type=str, choices=['mobilenetv2', 'resnet50', 'efficientnet'],
                        help='Model architecture to use (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--data-dir', type=str, help='Path to the dataset directory')
    parser.add_argument('--fine-tune', action='store_true', help='Enable fine-tuning')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    
    return parser.parse_args()

def setup_gpu(gpu_id=0):
    """Configure GPU settings."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the specified GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU: {gpus[gpu_id].name}")
        except RuntimeError as e:
            print(f"Error setting up GPU: {e}")
    else:
        print("No GPU available, using CPU.")

def train():
    """Main training function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up GPU
    setup_gpu(args.gpu)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.model:
        config['model']['name'] = args.model
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    
    # Create output directories
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(config_path=args.config)
    
    # Get data generators
    print("\nLoading and preprocessing data...")
    train_gen, val_gen = preprocessor.get_data_generators()
    
    # Save class indices
    class_indices_path = os.path.join(config['paths']['models_dir'], 'class_indices.json')
    with open(class_indices_path, 'w') as f:
        json.dump(train_gen.class_indices, f)
    print(f"\nClass indices saved to {class_indices_path}")
    
    # Initialize model builder
    print("\nInitializing model...")
    model_builder = ModelBuilder(config_path=args.config)
    
    # Build the model
    model, _ = model_builder.build_model()
    
    # Get callbacks
    callbacks = model_builder.get_callbacks()
    
    # Train the model
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    history = model.fit(
        train_gen,
        epochs=config['training']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Fine-tuning if enabled
    if args.fine_tune:
        print("\nStarting fine-tuning...")
        model = model_builder.fine_tune_model(model)
        
        # Train with fine-tuning
        history = model.fit(
            train_gen,
            epochs=config['training']['epochs'],
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
    
    # Save the final model
    final_model_path = os.path.join(
        config['paths']['models_dir'],
        f"{config['model']['name']}_final_model.h5"
    )
    model.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save training history
    history_path = os.path.join(
        config['paths']['models_dir'],
        f"{config['model']['name']}_training_history.json"
    )
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    print(f"Training history saved to {history_path}")
    
    # Evaluate the model
    print("\nEvaluating model...")
    evaluation = model.evaluate(val_gen)
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print("-" * 50)
    for name, value in zip(model.metrics_names, evaluation):
        print(f"{name}: {value:.4f}")
    
    return model, history

if __name__ == "__main__":
    # Start training
    start_time = datetime.now()
    print(f"Training started at {start_time}")
    print("=" * 80)
    
    try:
        model, history = train()
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print(f"Training completed at {end_time}")
        print(f"Total training time: {duration}")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        raise e
