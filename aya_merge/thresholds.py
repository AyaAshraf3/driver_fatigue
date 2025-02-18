# Thresholds for yawning and blinking
yawning_threshold= 2.0
microsleep_threshold= 3.0
eye_open_threshold= 0.30
mouth_closed_threshold=0.50

# Threshold for gaze and head pose detection
gaze_abnormal_duration = 5    # Duration (in seconds) to trigger abnormal gaze alert
head_abnormal_duration = 5    # Duration (in seconds) to trigger abnormal head movement alert
threshold_time = 15       # Duration in seconds to establish a baseline

# Thresholds for detecting abnormal head movements
PITCH_THRESHOLD = 10         # Angle in degrees for abnormal pitch
YAW_THRESHOLD = 10           # Angle in degrees for abnormal yaw
ROLL_THRESHOLD = 10          # Angle in degrees for abnormal roll
EAR_THRESHOLD = 0.35         # EAR threshold which below it considered looking down
NO_BLINK_GAZE_DURATION = 5  # Time (seconds) for center gaze without blinking to be considered abnormal
DISTRACTION_THRESHOLD=4