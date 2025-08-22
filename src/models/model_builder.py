"""
Model builder for the Smart Sorting System.
Defines and compiles the deep learning models for fruit/vegetable classification.
"""

import os
import yaml
import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers, callbacks
from tensorflow.keras.applications import (
    MobileNetV2, ResNet50, EfficientNetB0,
    mobilenet_v2, resnet, efficientnet
)
from tensorflow.keras.models import Model
from typing import Optional, Dict, Any, Tuple

class ModelBuilder:
    """Builds and compiles deep learning models for classification."""
    
    def __init__(self, config_path: str = '../../config/config.yaml'):
        """
        Initialize the model builder with configuration.
        
        Args:
            config_path: Path to the configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.input_shape = tuple(self.model_config['input_shape'])
        self.num_classes = self.model_config['num_classes']
        self.weights = self.model_config['weights']
        self.learning_rate = float(self.model_config['learning_rate'])
        self.dropout_rate = self.model_config['dropout_rate']
        
    def build_model(self, model_name: Optional[str] = None) -> tf.keras.Model:
        """
        Build a transfer learning model.
        
        Args:
            model_name: Name of the model architecture. If None, uses the one from config.
            
        Returns:
            A compiled Keras model.
        """
        if model_name is None:
            model_name = self.model_config['name'].lower()
            
        print(f"Building {model_name} model...")
        
        # Input layer
        input_tensor = layers.Input(shape=self.input_shape, name='input_layer')
        
        # Base model (pre-trained on ImageNet)
        base_model, preprocess_input = self._get_base_model(model_name, input_tensor)
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create new model on top
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        predictions = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output_layer'
        )(x)
        
        # Create the full model
        model = Model(
            inputs=input_tensor, 
            outputs=predictions,
            name=f"{model_name}_transfer"
        )
        
        # Compile the model
        self._compile_model(model)
        
        return model, preprocess_input
    
    def _get_base_model(self, model_name: str, input_tensor: tf.Tensor) -> Tuple[Model, callable]:
        """
        Get the base model and corresponding preprocessing function.
        
        Args:
            model_name: Name of the model architecture.
            input_tensor: Input tensor for the model.
            
        Returns:
            Tuple of (base_model, preprocess_input_function).
        """
        if model_name == 'mobilenetv2':
            base_model = MobileNetV2(
                weights=self.weights,
                include_top=False,
                input_tensor=input_tensor
            )
            preprocess_input = mobilenet_v2.preprocess_input
            
        elif model_name == 'resnet50':
            base_model = ResNet50(
                weights=self.weights,
                include_top=False,
                input_tensor=input_tensor
            )
            preprocess_input = resnet.preprocess_input
            
        elif model_name == 'efficientnet':
            base_model = EfficientNetB0(
                weights=self.weights,
                include_top=False,
                input_tensor=input_tensor
            )
            preprocess_input = efficientnet.preprocess_input
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
            
        return base_model, preprocess_input
    
    def _compile_model(self, model: tf.keras.Model) -> None:
        """
        Compile the model with appropriate loss, optimizer, and metrics.
        
        Args:
            model: Keras model to compile.
        """
        # Define optimizer
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)
        
        # Define loss function based on number of classes
        if self.num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
            
        # Compile the model
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print("Model compiled successfully!")
    
    def get_callbacks(self) -> list:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks.
        """
        # Create models directory if it doesn't exist
        os.makedirs(self.config['paths']['models_dir'], exist_ok=True)
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(
            self.config['paths']['models_dir'],
            f"{self.model_config['name']}_best_model.h5"
        )
        
        callbacks = [
            # Save the best model based on validation loss
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                mode='min',
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['patience'],
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=self.training_config['patience'] // 2,
                min_lr=1e-6,
                verbose=1
            ),
            
            # TensorBoard callback for visualization
            callbacks.TensorBoard(
                log_dir=os.path.join(self.config['paths']['logs_dir'], 'tensorboard'),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]
        
        return callbacks
    
    def fine_tune_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Fine-tune the model by unfreezing some layers.
        
        Args:
            model: The model to fine-tune.
            
        Returns:
            The fine-tuned model.
        """
        # Unfreeze the base model
        for layer in model.layers:
            if 'mobilenetv2' in layer.name or 'resnet50' in layer.name or 'efficientnet' in layer.name:
                layer.trainable = True
        
        # Recompile the model with a lower learning rate
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print("Model prepared for fine-tuning with lower learning rate.")
        return model


def main():
    """Test the model builder."""
    # Initialize model builder
    builder = ModelBuilder()
    
    # Build the model
    model, preprocess_input = builder.build_model()
    
    # Print model summary
    model.summary()
    
    # Get callbacks
    callbacks = builder.get_callbacks()
    print(f"\nUsing {len(callbacks)} callbacks:")
    for callback in callbacks:
        print(f"- {callback.__class__.__name__}")


if __name__ == "__main__":
    main()
