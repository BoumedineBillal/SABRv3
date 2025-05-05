# elevator_supervisor.py
# Supervisor that creates and controls the elevator inside a Solid inside a robot

from controller import Supervisor
import sys
import os

# Add path to import elevator_config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

try:
    from elevator_config import *
except ImportError:
    print("Error: Could not import elevator_config.py")
    # Default values if config can't be loaded
    ELEVATOR_WIDTH = 2.0
    ELEVATOR_LENGTH = 2.0
    ELEVATOR_HEIGHT = 0.1
    WALL_THICKNESS = 0.1
    WALL_HEIGHT = 2.0
    ELEVATOR_INITIAL_X = 0.0
    ELEVATOR_INITIAL_Y = 0.0
    ELEVATOR_INITIAL_Z = 0.0
    NUMBER_OF_FLOORS = 14
    FLOOR_HEIGHT_DIFFERENCE = 1.0
    ELEVATOR_COLOR = [0.7, 0.7, 0.7]
    WALL_COLOR = [0.5, 0.5, 0.5]
    ELEVATOR_SPEED = 1.0

class ElevatorSupervisor(Supervisor):
    def __init__(self):
        super(ElevatorSupervisor, self).__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # Initialize elevator state
        self.current_floor = 0
        self.target_floor = 0
        self.is_moving = False
        self.current_gesture = 0
        self.elevator_node = None
        self.trans_field = None
        self.running = True
        
        # Force initial position to be at origin
        self.elevator_position = [0.0, 0.0, 0.0]
        
        # Get or create the parent robot
        self.parent_robot = self.get_or_create_parent_robot()
        if not self.parent_robot:
            print("Error: Could not get or create parent robot")
            return
            
        # Get or create the Solid container inside the robot
        self.container_solid = self.get_or_create_container_solid()
        if not self.container_solid:
            print("Error: Could not get or create container Solid")
            return
        
        # Create the elevator as a child of the container Solid
        self.create_elevator()
        
        # Create floor indicators as children of the container Solid
        self.create_floor_indicators()
        
        # Set up keyboard for testing
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.timestep)
        
        print("Elevator supervisor initialized - Use number keys 0-9 to control the elevator")
    
    def get_or_create_parent_robot(self):
        """Get an existing robot or create a new one to serve as parent."""
        try:
            # First, try to find an existing robot named 'supervisor'
            robot_node = self.getFromDef("SUPERVISOR")
            if robot_node:
                print("Found existing SUPERVISOR robot")
                return robot_node
                
            # If not found, look for any robot
            # Check if the supervisor itself is a robot (most likely case)
            self_node = self.getSelf()
            if self_node:
                print("Using supervisor itself as parent robot")
                return self_node
                    
            # If no suitable robot found, return None
            print("Could not find a suitable parent robot")
            return None
        except Exception as e:
            print(f"Error in get_or_create_parent_robot: {e}")
            return None
    
    def get_or_create_container_solid(self):
        """Get or create a Solid node inside the parent robot to contain the elevator."""
        try:
            if not self.parent_robot:
                return None
                
            # Get the children field of the parent robot
            children_field = self.parent_robot.getField("children")
            if not children_field:
                print("Error: Could not get children field of parent robot")
                return None
                
            # First try to find an existing container
            container_node = self.getFromDef("ELEVATOR_CONTAINER")
            if container_node:
                print("Found existing ELEVATOR_CONTAINER Solid")
                return container_node
                
            # If not found, create a new container Solid
            container_def = """
            DEF ELEVATOR_CONTAINER Solid {
              children [
              ]
            }
            """
            
            # Add container to the parent robot's children
            children_field.importMFNodeFromString(-1, container_def)
            
            # Step the simulation to ensure the node is created
            self.step(self.timestep)
            
            # Get the container node
            container_node = self.getFromDef("ELEVATOR_CONTAINER")
            if container_node:
                print("Created new ELEVATOR_CONTAINER Solid")
                return container_node
            else:
                print("Error: Could not create ELEVATOR_CONTAINER Solid")
                return None
        except Exception as e:
            print(f"Error in get_or_create_container_solid: {e}")
            return None
    
    def create_elevator(self):
        """Create a simple elevator structure as a child of the container Solid."""
        try:
            if not self.container_solid:
                return
                
            # Get the children field of the container
            children_field = self.container_solid.getField("children")
            if not children_field:
                print("Error: Could not get children field of container Solid")
                return
            
            # Create a simple elevator (just a box with walls)
            elevator_def = f"""
            DEF ELEVATOR Transform {{
              translation 0 0 0
              children [
                DEF ELEVATOR_BODY Shape {{
                  appearance PBRAppearance {{
                    baseColor {ELEVATOR_COLOR[0]} {ELEVATOR_COLOR[1]} {ELEVATOR_COLOR[2]}
                    roughness 0.3
                    metalness 0.5
                  }}
                  geometry Box {{
                    size {ELEVATOR_WIDTH} {ELEVATOR_LENGTH} {ELEVATOR_HEIGHT}
                  }}
                }}
                DEF LEFT_WALL Transform {{
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
                DEF RIGHT_WALL Transform {{
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
                DEF BACK_WALL Transform {{
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
                DEF DISPLAY Transform {{
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
            
            # Add elevator to the container's children
            children_field.importMFNodeFromString(-1, elevator_def)
            
            # Step the simulation to ensure the node is created
            self.step(self.timestep)
            
            # Now try to get the elevator node
            self.elevator_node = self.getFromDef("ELEVATOR")
            if not self.elevator_node:
                print("Error: Could not get ELEVATOR node")
            else:
                # Get the translation field
                self.trans_field = self.elevator_node.getField("translation")
                if not self.trans_field:
                    print("Error: Could not get translation field")
                else:
                    print("Successfully initialized elevator node and translation field")
                    
                    # Force position to known good starting point
                    self.trans_field.setSFVec3f([0.0, 0.0, 0.0])
                    print("Set elevator position to origin [0.0, 0.0, 0.0]")
        except Exception as e:
            print(f"Error in create_elevator: {e}")
    
    def create_floor_indicators(self):
        """Create visual indicators for each floor as children of the container Solid."""
        try:
            if not self.container_solid:
                return
                
            # Get the children field of the container
            children_field = self.container_solid.getField("children")
            if not children_field:
                print("Error: Could not get children field of container Solid")
                return
            
            # Create a group to hold all floor indicators
            indicators_group_def = """
            DEF FLOOR_INDICATORS Group {
              children [
            """
            
            # Add indicator for each floor
            for floor in range(NUMBER_OF_FLOORS):
                floor_height = floor * FLOOR_HEIGHT_DIFFERENCE
                
                # Calculate color - make sure all values are in range [0, 1]
                color_intensity = min(1.0, floor / 13.0)
                green_value = max(0.0, 1.0 - (floor / 13.0))
                
                indicators_group_def += f"""
                DEF FLOOR_{floor}_INDICATOR Transform {{
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
            
            # Close the group definition
            indicators_group_def += """
              ]
            }
            """
            
            # Add indicators group to the container's children
            children_field.importMFNodeFromString(-1, indicators_group_def)
            
            # Step the simulation to ensure nodes are created
            self.step(self.timestep)
        except Exception as e:
            print(f"Error in create_floor_indicators: {e}")
    
    def is_node_valid(self, node):
        """Check if a node is still valid."""
        if node is None:
            return False
        try:
            # Try to access some property of the node
            # If this fails, the node is no longer valid
            node.getId()
            return True
        except Exception:
            return False
    
    def move_elevator(self):
        """Move the elevator towards the target floor."""
        try:
            if not self.is_node_valid(self.elevator_node) or not self.trans_field:
                print("Cannot move elevator: elevator_node is no longer valid or trans_field is None")
                return
            
            # Calculate target height and current height
            target_height = self.target_floor * FLOOR_HEIGHT_DIFFERENCE
            
            # Get current position directly from our tracking variable
            current_position = self.elevator_position
            current_height = current_position[2]
            
            # Calculate direction and movement
            height_difference = target_height - current_height
            if abs(height_difference) < 0.01:
                # Reached target floor
                self.current_floor = self.target_floor
                self.is_moving = False
                print(f"Arrived at floor {self.current_floor}")
            else:
                # Move towards target floor
                direction = 1 if height_difference > 0 else -1
                movement = min(abs(height_difference), ELEVATOR_SPEED * self.timestep / 1000.0) * direction
                
                # Update position
                new_position = [current_position[0], current_position[1], current_height + movement]
                
                # Store the position in our tracking variable
                self.elevator_position = new_position
                
                # Update the actual elevator position
                self.trans_field.setSFVec3f(new_position)
        except Exception as e:
            print(f"Error in move_elevator: {e}")
            self.is_moving = False
    
    def set_target_floor(self, floor):
        """Set the target floor."""
        if 0 <= floor < NUMBER_OF_FLOORS and floor != self.current_floor:
            self.target_floor = floor
            self.is_moving = True
            print(f"Moving to floor {floor}")
    
    def set_current_gesture(self, gesture):
        """Set the current recognized gesture and move to corresponding floor."""
        if 0 <= gesture < NUMBER_OF_FLOORS:
            self.current_gesture = gesture
            self.set_target_floor(gesture)
    
    def display_status(self):
        """Display current status in the console."""
        if self.getTime() % 1.0 < 0.1 or self.is_moving:
            status = (f"Current floor: {self.current_floor}, "
                     f"Target floor: {self.target_floor}, "
                     f"Moving: {'Yes' if self.is_moving else 'No'}, "
                     f"Current gesture: {self.current_gesture}")
            print(status)
    
    def run(self):
        """Main control loop."""
        while self.step(self.timestep) != -1 and self.running:
            try:
                # Check if nodes are still valid
                if not self.is_node_valid(self.parent_robot) or not self.is_node_valid(self.container_solid):
                    print("Warning: parent_robot or container_solid is no longer valid")
                    self.running = False
                    break
                
                # Process keyboard input (for testing)
                key = self.keyboard.getKey()
                if key != -1:
                    # Convert ASCII value to number
                    if 48 <= key <= 57:  # ASCII values for keys 0-9
                        floor = key - ord('0')
                        if 0 <= floor < NUMBER_OF_FLOORS:
                            self.set_target_floor(floor)
                            print(f"Key pressed: {floor}")
                
                # Move elevator if needed
                if self.is_moving:
                    self.move_elevator()
                    
                # Display current status
                self.display_status()
            except Exception as e:
                print(f"Error in main loop: {e}")
                self.running = False
                break
        
        print("Elevator supervisor shutting down cleanly")

# Initialize the supervisor
try:
    supervisor = ElevatorSupervisor()
    supervisor.run()
except Exception as e:
    print(f"Error in supervisor initialization or run: {e}")
