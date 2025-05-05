# gesture_recognition.py
# Module for hand gesture recognition in the elevator simulation

import os
import cv2
import numpy as np
import random
import time
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageOps

# Class names for the 14 gestures
GESTURE_CLASSES = [f"Gesture_{i}" for i in range(14)]

class GestureRecognizer:
    """Class for recognizing hand gestures from images."""
    
    def __init__(self, model_path=None, dataset_path=None, image_size=128):
        """
        Initialize the gesture recognizer.
        
        Args:
            model_path: Path to the trained model file
            dataset_path: Path to the dataset directory containing gesture folders
            image_size: Size of the input images for the model
        """
        self.image_size = image_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self.get_vgg_model(num_classes=14)
        
        # If model path is specified, load the model
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Model loaded from {model_path}")
        else:
            print("Warning: No model path specified or model file not found.")
            print("Using uninitialized model.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Load sample images from the dataset
        self.sample_images = {}
        if dataset_path and os.path.exists(dataset_path):
            self.load_sample_images(dataset_path)
        
        # Transform pipeline for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_vgg_model(self, num_classes=14, pretrained=False):
        """Get a VGG11 model with a custom classifier."""
        model = models.vgg11(pretrained=pretrained)
        
        # Replace the classifier with our custom sequence
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # No dropout for inference
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)
        )
        
        return model
    
    def load_sample_images(self, dataset_dir, size=150):
        """
        Load sample images for each gesture class from the dataset.
        
        Args:
            dataset_dir: Path to the dataset directory containing gesture folders
            size: Size to resize the sample images to
        """
        print(f"Loading sample images from {dataset_dir}...")
        
        # For each gesture class
        for gesture_id in range(14):
            gesture_dir = os.path.join(dataset_dir, f"Gesture_{gesture_id}")
            
            # Check if gesture directory exists
            if not os.path.exists(gesture_dir):
                print(f"Warning: Directory for Gesture_{gesture_id} not found.")
                continue
            
            # Get list of image files in the gesture directory
            image_files = [f for f in os.listdir(gesture_dir) if f.endswith('.jpg')]
            
            if not image_files:
                print(f"Warning: No image files found for Gesture_{gesture_id}.")
                continue
            
            # Randomly select one image from the class
            sample_file = random.choice(image_files)
            sample_path = os.path.join(gesture_dir, sample_file)
            
            try:
                # Use PIL for EXIF orientation handling
                pil_img = Image.open(sample_path).convert('RGB')
                pil_img = ImageOps.exif_transpose(pil_img)
                
                # Convert to OpenCV format for visualization
                sample_img = np.array(pil_img)
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR)
                
                # Resize the image
                sample_img = cv2.resize(sample_img, (size, size))
                
                # Add a label
                cv2.putText(sample_img, f"Gesture {gesture_id}", (5, 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                self.sample_images[gesture_id] = sample_img
                print(f"Loaded sample for Gesture_{gesture_id}")
            except Exception as e:
                print(f"Error loading sample for Gesture_{gesture_id}: {str(e)}")
        
        print(f"Loaded {len(self.sample_images)} sample images.")
    
    def preprocess_image(self, image):
        """
        Preprocess an image for input to the model.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Tensor for model input
        """
        # Convert BGR (OpenCV format) to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Apply transform
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image):
        """
        Predict the gesture class from an image.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Predicted gesture class (0-13)
        """
        # Preprocess the image
        input_tensor = self.preprocess_image(image)
        input_tensor = input_tensor.to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction = torch.argmax(probabilities).item()
            confidence = probabilities[prediction].item()
        
        return prediction, confidence
    
    def get_sample_image(self, gesture_id):
        """
        Get a sample image for a gesture class.
        
        Args:
            gesture_id: Gesture class ID (0-13)
            
        Returns:
            Sample image for the gesture class
        """
        if gesture_id in self.sample_images:
            return self.sample_images[gesture_id]
        else:
            return None
