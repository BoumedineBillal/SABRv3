# elevator_config.py
# Configuration parameters for the elevator simulation

# Elevator dimensions
ELEVATOR_WIDTH = 2.0
ELEVATOR_LENGTH = 2.0
ELEVATOR_HEIGHT = 0.1

# Wall dimensions
WALL_THICKNESS = 0.1
WALL_HEIGHT = 2.0

# Elevator position
ELEVATOR_INITIAL_X = 0.0
ELEVATOR_INITIAL_Y = 0.0
ELEVATOR_INITIAL_Z = 0.0

# Floors configuration
NUMBER_OF_FLOORS = 14  # Corresponding to gestures 0-13
FLOOR_HEIGHT_DIFFERENCE = 2.0  # Height between floors

# Colors
ELEVATOR_COLOR = [0.7, 0.7, 0.7]  # Gray
WALL_COLOR = [0.5, 0.5, 0.5]  # Darker gray

# Elevator movement
ELEVATOR_SPEED = 1.0  # Units per second
