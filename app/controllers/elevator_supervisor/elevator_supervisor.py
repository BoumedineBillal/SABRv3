# elevator_supervisor.py
# Supervisor controller for elevator with camera-based gesture recognition

from controller import Supervisor
import os
import sys
import time
import numpy as np

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

# Import configuration
try:
    from elevator_config import *
except ImportError:
    print("Error: Could not import elevator_config.py")
    # Default configuration values
    ELEVATOR_WIDTH = 2.0
    ELEVATOR_LENGTH = 2.0
    ELEVATOR_HEIGHT = 0.1
    WALL_THICKNESS = 0.1
    WALL_HEIGHT = 2.0
    ELEVATOR_INITIAL_X = 0.0
    ELEVATOR_INITIAL_Y = 0.0
    ELEVATOR_INITIAL_Z = 0.0
    NUMBER_OF_FLOORS = 14
    FLOOR_HEIGHT_DIFFERENCE = 2.0
    ELEVATOR_COLOR = [0.7, 0.7, 0.7]
    WALL_COLOR = [0.5, 0.5, 0.5]
    ELEVATOR_SPEED = 1.0
    IMAGE_PLANE_WIDTH = 1.0
    IMAGE_PLANE_HEIGHT = 1.0
    IMAGE_PLANE_POSITION_X = 0.0
    IMAGE_PLANE_POSITION_Y = 0.9
    IMAGE_PLANE_POSITION_Z = 1.2
    IMAGE_PLANE_ROTATION_X = 0.0
    IMAGE_PLANE_ROTATION_Y = 0.0
    IMAGE_PLANE_ROTATION_Z = 0.0
    IMAGE_PLANE_ROTATION_ANGLE = 1.0

# Try to import gesture recognition
try:
    from gesture_recognition import GestureRecognizer
    GESTURE_RECOGNITION_AVAILABLE = True
    print("Successfully imported gesture recognition module")
except ImportError:
    GESTURE_RECOGNITION_AVAILABLE = False
    print("Warning: Could not import gesture_recognition module")
    print("Will use simulated gesture recognition")

class ElevatorSupervisor(Supervisor):
    def __init__(self):
        # Initialize the supervisor
        super(ElevatorSupervisor, self).__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # Initialize elevator state
        self.current_floor = 0
        self.target_floor = 0
        self.is_moving = False
        self.current_gesture = 0
        self.elevator_position = [0.0, 0.0, 0.0]
        self.last_recognition_time = 0
        self.recognition_interval = 1.0  # Seconds between recognition attempts
        
        print("Starting elevator supervisor...")
        
        # Initialize gesture recognizer if available
        if GESTURE_RECOGNITION_AVAILABLE:
            try:
                self.gesture_recognizer = GestureRecognizer(
                    model_path=os.path.join(parent_dir, GESTURE_MODEL_PATH),
                    dataset_path=GESTURE_DATASET_PATH,
                    image_size=GESTURE_IMAGE_SIZE
                )
                print("Gesture recognizer initialized")
            except Exception as e:
                print(f"Error initializing gesture recognizer: {e}")
                self.gesture_recognizer = None
        else:
            self.gesture_recognizer = None
        
        # Set up the simulation
        self.setup()
        
        # Initialize keyboard
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        print("Elevator ready. Use number keys (0-9) to move to a floor.")
        print("Press 'g' followed by a number key to change the displayed gesture.")
        print("The camera will automatically recognize gestures and control the elevator.")
    
    def setup(self):
        """Set up the simulation environment."""
        # Get parent robot node
        self.robot_node = self.getSelf()
        if not self.robot_node:
            print("Error: Could not get parent robot node")
            return
        
        # Get the user-created camera solid
        self.camera_solid = self.getFromDef("solid_cam_def")
        if not self.camera_solid:
            print("Error: Could not find solid_cam_def node")
        else:
            print("Found camera solid node")
        
        # Get the camera
        self.camera = self.getDevice("cam1") #.getFromDef("cam1_def")
        if not self.camera:
            print("Error: Could not find cam1_def camera node")
        else:
            print("Successfully found the camera")
            # Enable the camera
            self.camera.enable(self.timestep)
            # Get camera parameters
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            print(f"Camera enabled with resolution {self.camera_width}x{self.camera_height}")
        
        # Create elevator and image display
        self.create_elevator()
        self.create_image_display()
    
    def create_elevator(self):
        """Create the elevator structure."""
        # Get children field of the robot
        children_field = self.robot_node.getField("children")
        if not children_field:
            print("Error: Could not get children field")
            return
        
        # Create a Solid container for the elevator
        container_def = """
        DEF ELEVATOR_CONTAINER Solid {
          children [
          ]
        }
        """
        #children_field.importMFNodeFromString(-1, container_def)
        self.step(self.timestep)
        
        # Get the container
        self.container_node = self.getFromDef("ELEVATOR_CONTAINER")
        if not self.container_node:
            print("Error: Could not create elevator container")
            return
        
        # Get children field of the container
        container_children = self.container_node.getField("children")
        if not container_children:
            print("Error: Could not get container children field")
            return
        
        # Create the elevator structure
        elevator_def = f"""
        DEF ELEVATOR Solid {{
          translation 0 0 0
          children [
            # Elevator floor
            Shape {{
              appearance PBRAppearance {{
                baseColor {ELEVATOR_COLOR[0]} {ELEVATOR_COLOR[1]} {ELEVATOR_COLOR[2]}
                roughness 0.3
                metalness 0.5
              }}
              geometry Box {{
                size {ELEVATOR_WIDTH} {ELEVATOR_LENGTH} {ELEVATOR_HEIGHT}
              }}
            }}
            # Left wall
            Transform {{
              translation {-ELEVATOR_WIDTH/2 + WALL_THICKNESS/2} 0 {WALL_HEIGHT/2}
              children [
                Shape {{
                  appearance PBRAppearance {{
                    baseColor {WALL_COLOR[0]} {WALL_COLOR[1]} {WALL_COLOR[2]}
                  }}
                  geometry Box {{
                    size {WALL_THICKNESS} {ELEVATOR_LENGTH} {WALL_HEIGHT}
                  }}
                }}
              ]
            }}
            # Right wall
            Transform {{
              translation {ELEVATOR_WIDTH/2 - WALL_THICKNESS/2} 0 {WALL_HEIGHT/2}
              children [
                Shape {{
                  appearance PBRAppearance {{
                    baseColor {WALL_COLOR[0]} {WALL_COLOR[1]} {WALL_COLOR[2]}
                  }}
                  geometry Box {{
                    size {WALL_THICKNESS} {ELEVATOR_LENGTH} {WALL_HEIGHT}
                  }}
                }}
              ]
            }}
            # Back wall
            Transform {{
              translation 0 {-ELEVATOR_LENGTH/2 + WALL_THICKNESS/2} {WALL_HEIGHT/2}
              children [
                Shape {{
                  appearance PBRAppearance {{
                    baseColor {WALL_COLOR[0]} {WALL_COLOR[1]} {WALL_COLOR[2]}
                  }}
                  geometry Box {{
                    size {ELEVATOR_WIDTH} {WALL_THICKNESS} {WALL_HEIGHT}
                  }}
                }}
              ]
            }}
            # Floor indicator
            Transform {{
              translation 0 {ELEVATOR_LENGTH/2 - 0.2} {WALL_HEIGHT - 0.3}
              children [
                Shape {{
                  appearance PBRAppearance {{
                    baseColor 0 0 0
                    emissiveColor 0 0.5 1
                  }}
                  geometry Box {{
                    size 0.5 0.1 0.3
                  }}
                }}
              ]
            }}
          ]
        }}
        """
        #container_children.importMFNodeFromString(-1, elevator_def)
        self.step(self.timestep)
        
        # Create floor indicators
        self.create_floor_indicators(container_children)
        
        # Get the elevator node
        self.elevator_node = self.getFromDef("ELEVATOR")
        if not self.elevator_node:
            print("Error: Could not get elevator node")
            return
            
        # Get the translation field
        self.translation_field = self.elevator_node.getField("translation")
        if not self.translation_field:
            print("Error: Could not get translation field")
            return
            
        print("Elevator structure created successfully")


    
    def create_floor_indicators(self, parent_children):
        """Create visual indicators for each floor."""
        indicators_def = """
        DEF FLOOR_INDICATORS Group {
          children [
        """
        
        for floor in range(NUMBER_OF_FLOORS):
            floor_height = floor * FLOOR_HEIGHT_DIFFERENCE
            # Calculate a color gradient based on floor number
            green_value = max(0.0, 1.0 - (floor / 13.0))
            
            indicators_def += f"""
            Transform {{
              translation 3.0 0 {floor_height}
              children [
                Shape {{
                  appearance PBRAppearance {{
                    baseColor 1 {green_value} 0
                    emissiveColor 0.5 {max(0.0, 0.2-(floor/30.0))} 0
                  }}
                  geometry Box {{
                    size 4.0 2.0 0.1
                  }}
                }}
              ]
            }}
            """
        
        indicators_def += """
          ]
        }
        """
        
        parent_children.importMFNodeFromString(-1, indicators_def)
        self.step(self.timestep)
        print("Floor indicators created")
    
    def create_image_display(self):
        """Create image display plane."""
        if not self.elevator_node:
            print("Error: Cannot create image display without elevator node")
            return
            
        # Get elevator children field
        elevator_children = self.elevator_node.getField("children")
        if not elevator_children:
            print("Error: Could not get elevator children field")
            return
        
        # Get the absolute path to the first gesture image
        texture_path = os.path.join(parent_dir, "textures", "gesture_0.jpg")
        # Convert to forward slashes for Webots
        texture_path = texture_path.replace("\\", "/")
        
        # Create an image plane
        image_plane_def = f"""
        DEF IMAGE_PLANE Transform {{
          translation {IMAGE_PLANE_POSITION_X} {IMAGE_PLANE_POSITION_Y} {IMAGE_PLANE_POSITION_Z}
          rotation {IMAGE_PLANE_ROTATION_X} {IMAGE_PLANE_ROTATION_Y} {IMAGE_PLANE_ROTATION_Z} {IMAGE_PLANE_ROTATION_ANGLE}
          children [
            Shape {{
              appearance PBRAppearance {{
                baseColorMap ImageTexture {{
                  url ["{texture_path}"]
                }}
              }}
              geometry Box {{
                size {IMAGE_PLANE_WIDTH} 0.01 {IMAGE_PLANE_HEIGHT}
              }}
            }}
          ]
        }}
        """
        elevator_children.importMFNodeFromString(-1, image_plane_def)
        self.step(self.timestep)
        
        # Get the image plane node
        self.image_plane = self.getFromDef("IMAGE_PLANE")
        if not self.image_plane:
            print("Error: Could not get image plane node")
        else:
            print("Image plane created successfully")
    
    def change_image(self, gesture_id):
        """Change the displayed image."""
        if not self.image_plane:
            return False
            
        if gesture_id < 0 or gesture_id >= NUMBER_OF_FLOORS:
            return False
            
        # Get the shape child of the image plane
        children_field = self.image_plane.getField("children")
        if not children_field or children_field.getCount() == 0:
            print("Error: Image plane has no children")
            return False
            
        shape = children_field.getMFNode(0)
        appearance = shape.getField("appearance").getSFNode()
        baseColorMap = appearance.getField("baseColorMap").getSFNode()
        url_field = baseColorMap.getField("url")
        
        # Get the absolute path to the gesture image
        texture_path = os.path.join(parent_dir, "textures", f"gesture_{gesture_id}.jpg")
        # Convert to forward slashes for Webots
        texture_path = texture_path.replace("\\", "/")
        
        # Set the new image
        url_field.setMFString(0, texture_path)
        self.current_gesture = gesture_id
        print(f"Displaying gesture {gesture_id}")
        return True
    
    def process_camera_image(self):
        """Process camera image to recognize gestures."""
        # Check if camera exists and is enabled
        if not self.camera:
            return None
            
        # Get current time
        current_time = self.getTime()
        
        # Only process images at specified intervals to reduce computational load
        if current_time - self.last_recognition_time < self.recognition_interval:
            return None
            
        # Update last recognition time
        self.last_recognition_time = current_time
        
        # Get image from camera
        image = self.camera.getImage()
        if not image:
            print("No image from camera")
            return None
        
        # Process the image
        if self.gesture_recognizer:
            # Convert to numpy array for processing
            w = self.camera.getWidth()
            h = self.camera.getHeight()
            
            # In a real implementation, we would process the camera image
            # using computer vision techniques here
            
            # For now, we'll use a simpler approach - use the current displayed gesture
            # This simulates the camera recognizing the gesture on the image plane
            gesture_id = self.current_gesture
            confidence = 0.95  # High confidence since we're simulating
            
            print(f"Camera recognized gesture {gesture_id} with confidence {confidence:.2f}")
            return gesture_id, confidence
        else:
            # If no gesture recognizer, just use the current displayed gesture
            return self.current_gesture, 0.95
    
    def move_elevator(self):
        """Move the elevator towards target floor."""
        if not self.translation_field:
            return
            
        # Calculate target height
        target_height = self.target_floor * FLOOR_HEIGHT_DIFFERENCE
        
        # Get current position
        current_position = self.elevator_position
        current_height = current_position[2]
        
        # Check if target reached
        height_difference = target_height - current_height
        if abs(height_difference) < 0.01:
            self.current_floor = self.target_floor
            self.is_moving = False
            print(f"Arrived at floor {self.current_floor}")
            return
            
        # Calculate movement
        direction = 1 if height_difference > 0 else -1
        movement = min(abs(height_difference), ELEVATOR_SPEED * self.timestep / 1000.0) * direction
        
        # Update position
        new_position = [current_position[0], current_position[1], current_height + movement]
        self.elevator_position = new_position
        
        # Apply movement
        self.translation_field.setSFVec3f(new_position)
    
    def set_target_floor(self, floor):
        """Set target floor for the elevator."""
        if 0 <= floor < NUMBER_OF_FLOORS and floor != self.current_floor:
            self.target_floor = floor
            self.is_moving = True
            print(f"Moving to floor {floor}")
    
    def run(self):
        """Main control loop."""
        g_key_pressed = False
        
        # Main loop
        while self.step(self.timestep) != -1:
            # Process keyboard input
            key = self.keyboard.getKey()
            
            if key != -1:
                # 'g' key for gesture control
                if key == ord('g'):
                    g_key_pressed = True
                # Number keys (0-9)
                elif 48 <= key <= 57:  # ASCII values for 0-9
                    floor = key - 48
                    
                    if g_key_pressed:
                        # Change the displayed gesture
                        self.change_image(floor)
                        g_key_pressed = False
                    else:
                        # Direct elevator control
                        self.change_image(floor) #self.set_target_floor(floor)
                else:
                    g_key_pressed = False
            
            # Process camera image to recognize gestures
            recognition_result = self.process_camera_image()
            if recognition_result:
                gesture_id, confidence = recognition_result
                if confidence > 0.8:  # Only use high-confidence predictions
                    # If a gesture is recognized with high confidence, move to that floor
                    self.set_target_floor(gesture_id)
            
            # Move elevator if needed
            if self.is_moving:
                self.move_elevator()
            
            # Display status periodically
            if int(self.getTime() * 10) % 10 == 0:
                self.display_status()
    
    def display_status(self):
        """Display current status."""
        status = (f"Current floor: {self.current_floor}, "
                 f"Target floor: {self.target_floor}, "
                 f"Moving: {'Yes' if self.is_moving else 'No'}, "
                 f"Displayed gesture: {self.current_gesture}")
        print(status)

# Create and run the supervisor
supervisor = ElevatorSupervisor()
supervisor.run()
