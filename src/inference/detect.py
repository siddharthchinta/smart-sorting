"""
Inference script for the Smart Sorting System.
Loads a trained model and performs inference on new images.
"""

import os
import sys
import yaml
import json
import numpy as np
import tensorflow as tf
import cv2
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Union

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

class FruitVegetableDetector:
    """Class for detecting and classifying fruits/vegetables as fresh or rotten."""
    
    def __init__(self, config_path: str = '../../config/config.yaml', model_path: str = None):
        """
        Initialize the detector with configuration and model.
        
        Args:
            config_path: Path to the configuration file.
            model_path: Path to the trained model. If None, looks in the default models directory.
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up paths
        self.models_dir = self.config['paths']['models_dir']
        self.input_shape = tuple(self.config['model']['input_shape'])
        
        # Set default model path if not provided
        if model_path is None:
            model_name = self.config['model']['name']
            model_path = os.path.join(self.models_dir, f"{model_name}_final_model.h5")
            
        # Load class indices
        class_indices_path = os.path.join(self.models_dir, 'class_indices.json')
        with open(class_indices_path, 'r') as f:
            self.class_indices = json.load(f)
            self.class_names = {v: k for k, v in self.class_indices.items()}
            
        # Load the model
        print(f"Loading model from {model_path}...")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.compile(optimizer='adam', 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy'])
        
        # Set up preprocessing function based on model type
        self._setup_preprocessing()
        
        print("Model loaded successfully!")
        print(f"Classes: {self.class_indices}")
    
    def _setup_preprocessing(self):
        """Set up the appropriate preprocessing function based on the model type."""
        model_name = self.config['model']['name'].lower()
        
        if 'mobilenet' in model_name:
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        elif 'resnet' in model_name:
            from tensorflow.keras.applications.resnet50 import preprocess_input
        elif 'efficientnet' in model_name:
            from tensorflow.keras.applications.efficientnet import preprocess_input
        else:
            # Default to no preprocessing
            preprocess_input = lambda x: x / 255.0
            
        self.preprocess_input = preprocess_input
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess an image for inference.
        
        Args:
            image: Input image as a NumPy array (BGR format from OpenCV).
            
        Returns:
            Preprocessed image as a NumPy array.
        """
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, (self.input_shape[1], self.input_shape[0]))
        
        # Apply model-specific preprocessing
        image = self.preprocess_input(image)
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image as a NumPy array (BGR format from OpenCV).
            
        Returns:
            Tuple of (class_name, confidence).
        """
        # Preprocess the image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get the predicted class and confidence
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        class_name = self.class_names[class_idx]
        
        return class_name, confidence
    
    def process_image_file(self, image_path: str) -> Dict[str, Union[str, float]]:
        """
        Process an image file and return the prediction.
        
        Args:
            image_path: Path to the input image file.
            
        Returns:
            Dictionary containing the prediction results.
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        # Make prediction
        class_name, confidence = self.predict(image)
        
        return {
            'image_path': image_path,
            'class_name': class_name,
            'confidence': confidence,
            'is_fresh': class_name.lower() == 'fresh'
        }
    
    def process_video(self, video_path: str, output_path: str = None, 
                     display: bool = False, confidence_threshold: float = 0.7) -> None:
        """
        Process a video file and save/display the results.
        
        Args:
            video_path: Path to the input video file.
            output_path: Path to save the output video. If None, the video is not saved.
            display: Whether to display the video in real-time.
            confidence_threshold: Minimum confidence threshold for displaying predictions.
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total Frames: {total_frames}")
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Make a copy of the frame for display
            display_frame = frame.copy()
            
            try:
                # Make prediction
                class_name, confidence = self.predict(frame)
                
                # Only display predictions above the confidence threshold
                if confidence >= confidence_threshold:
                    # Draw prediction on the frame
                    label = f"{class_name}: {confidence:.2f}"
                    color = (0, 255, 0) if class_name.lower() == 'fresh' else (0, 0, 255)
                    
                    cv2.putText(display_frame, label, (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    
                    # Draw a rectangle around the detected object
                    h, w = frame.shape[:2]
                    cv2.rectangle(display_frame, (0, 0), (w, h), color, 2)
                
                # Write the frame to the output video
                if output_path:
                    out.write(display_frame)
                    
                # Display the frame
                if display:
                    cv2.imshow('Smart Sorting System', display_frame)
                    
                    # Press 'q' to quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                
            frame_count += 1
            print(f"Processed frame {frame_count}/{total_frames}", end='\r')
        
        # Release resources
        cap.release()
        if output_path:
            out.release()
        if display:
            cv2.destroyAllWindows()
            
        print("\nVideo processing complete!")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect and classify fruits/vegetables as fresh or rotten.')
    parser.add_argument('--image', type=str, help='Path to the input image file')
    parser.add_argument('--video', type=str, help='Path to the input video file')
    parser.add_argument('--output', type=str, help='Path to save the output video file')
    parser.add_argument('--display', action='store_true', help='Display the output in real-time')
    parser.add_argument('--config', type=str, default='../../config/config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--model', type=str, help='Path to the trained model')
    parser.add_argument('--confidence', type=float, default=0.7,
                        help='Minimum confidence threshold for detection (0-1)')
    
    return parser.parse_args()

def main():
    """Main function for running inference."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize the detector
    detector = FruitVegetableDetector(config_path=args.config, model_path=args.model)
    
    # Process image if provided
    if args.image:
        try:
            result = detector.process_image_file(args.image)
            print("\nPrediction Results:")
            print("-" * 50)
            print(f"Image: {result['image_path']}")
            print(f"Class: {result['class_name']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Is Fresh: {result['is_fresh']}")
            
            # Display the image with prediction
            image = cv2.imread(args.image)
            label = f"{result['class_name']}: {result['confidence']:.2f}"
            color = (0, 255, 0) if result['is_fresh'] else (0, 0, 255)
            
            cv2.putText(image, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Smart Sorting System', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    # Process video if provided
    elif args.video:
        try:
            detector.process_video(
                video_path=args.video,
                output_path=args.output,
                display=args.display,
                confidence_threshold=args.confidence
            )
        except Exception as e:
            print(f"Error processing video: {str(e)}")
    
    else:
        print("Please provide either --image or --video argument.")
        print("Use --help for more information.")

if __name__ == "__main__":
    main()
