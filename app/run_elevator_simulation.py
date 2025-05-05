# run_elevator_simulation.py
# Script to run the elevator simulation with gesture recognition

import os
import sys
import subprocess
import time

def main():
    """Main function to run the elevator simulation."""
    print("Starting elevator simulation with gesture recognition...")
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Run the gesture image preparation script
    print("\nPreparing gesture images...")
    try:
        subprocess.run([sys.executable, os.path.join(current_dir, "prepare_gesture_images.py")], check=True)
        print("Gesture images prepared successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error preparing gesture images: {e}")
        return
    
    # Start Webots with the elevator world
    print("\nStarting Webots with elevator simulation...")
    try:
        # The path to Webots and the world file may need to be adjusted for your system
        webots_path = "C:\\Program Files\\Webots\\webots.exe"  # Adjust this path
        world_path = os.path.join(current_dir, "worlds", "app.wbt")
        
        if os.path.exists(webots_path) and os.path.exists(world_path):
            subprocess.Popen([webots_path, world_path])
            print(f"Started Webots with {world_path}")
        else:
            print(f"Error: Webots executable or world file not found.")
            print(f"Webots path: {webots_path}")
            print(f"World path: {world_path}")
    except Exception as e:
        print(f"Error starting Webots: {e}")
    
    print("\nInstructions:")
    print("- Press number keys 0-9 to directly control the elevator")
    print("- Press 'g' followed by a number key to change the displayed gesture image")
    print("- The elevator will automatically move to the floor corresponding to the recognized gesture")

if __name__ == "__main__":
    main()
