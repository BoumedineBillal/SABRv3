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

# Camera configuration
CAMERA_POSITION_X = 0.0
CAMERA_POSITION_Y = 0.8  # Position near front of elevator
CAMERA_POSITION_Z = 1.2  # Height of camera
CAMERA_ROTATION_X = 0.0  # Rotation around X axis (tilt down)
CAMERA_ROTATION_Y = 1.57  # Rotation around Y axis (90 degrees to face image plane)
CAMERA_ROTATION_Z = 0.0
CAMERA_FIELD_OF_VIEW = 0.8
CAMERA_WIDTH = 320  # Camera resolution width
CAMERA_HEIGHT = 240  # Camera resolution height

# Image plane configuration
IMAGE_PLANE_WIDTH = 1.0
IMAGE_PLANE_HEIGHT = 1.0
IMAGE_PLANE_POSITION_X = -0.508417
IMAGE_PLANE_POSITION_Y = 0.733807  # Position on front wall -0.508417 , 0.733807 , 0.88
IMAGE_PLANE_POSITION_Z = 0.88  # Height on wall
IMAGE_PLANE_ROTATION_X = 0.0
IMAGE_PLANE_ROTATION_Y = 0.0
IMAGE_PLANE_ROTATION_Z = -1.0
IMAGE_PLANE_ROTATION_ANGLE = 1.618   # in radians

# Gesture recognition configuration
GESTURE_MODEL_PATH = r"C:\Users\orani\bilel\a_miv\a_miv\m1s2\rnna\tp\project\SABRv3\notebook_results\pruned_model\pruned_model_std_based_teta1_0.2_gamma_0.1.pth"
GESTURE_DATASET_PATH = "C:/Users/orani/bilel/a_miv/a_miv/m1s2/rnna/tp/project/deep_sp_smart_l1sh/dataset/HG14/HG14-Hand Gesture"
GESTURE_IMAGE_SIZE = 128  # Size of input images for the model
