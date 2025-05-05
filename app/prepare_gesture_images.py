# prepare_gesture_images.py
# Script to prepare gesture images for the elevator simulation

import os
import shutil
import random
from PIL import Image, ImageDraw, ImageFont
import sys

# Add path to import elevator_config
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from elevator_config import *
except ImportError:
    print("Error: Could not import elevator_config.py")
    # Set default values
    GESTURE_DATASET_PATH = "C:/Users/orani/bilel/a_miv/a_miv/m1s2/rnna/tp/project/deep_sp_smart_l1sh/dataset/HG14/HG14-Hand Gesture"

def prepare_gesture_images():
    """
    Prepare gesture images for the elevator simulation.
    Copies sample images from each gesture class to the textures directory.
    Adds a label to each image indicating the gesture class.
    """
    print("Preparing gesture images...")
    
    # Create textures directory if it doesn't exist
    textures_dir = os.path.join(current_dir, "textures")
    os.makedirs(textures_dir, exist_ok=True)
    
    # For each gesture class
    for gesture_id in range(14):
        gesture_dir = os.path.join(GESTURE_DATASET_PATH, f"Gesture_{gesture_id}")
        
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
        
        # Destination path in textures directory
        dest_path = os.path.join(textures_dir, f"gesture_{gesture_id}.jpg")
        
        try:
            # Open the image with PIL
            img = Image.open(sample_path)
            
            # Resize the image to a standard size
            img = img.resize((512, 512))
            
            # Add a label to the image
            draw = ImageDraw.Draw(img)
            try:
                # Try to use a specific font
                font = ImageFont.truetype("arial.ttf", 36)
            except IOError:
                # If the font is not available, use the default font
                font = ImageFont.load_default()
            
            # Draw a semi-transparent background for the text
            draw.rectangle([(10, 10), (300, 60)], fill=(0, 0, 0, 128))
            
            # Draw the text
            draw.text((20, 20), f"Gesture {gesture_id}", fill=(255, 255, 255), font=font)
            
            # Save the image
            img.save(dest_path)
            print(f"Saved {dest_path}")
        except Exception as e:
            print(f"Error preparing image for Gesture_{gesture_id}: {e}")
    
    print("Gesture images preparation complete.")

if __name__ == "__main__":
    prepare_gesture_images()
