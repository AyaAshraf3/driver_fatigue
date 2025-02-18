import cv2
import torch
import time

# Load YOLOv5 model (ensure you have the correct weights path)
'''
torch.hub.load(): A function in PyTorch that loads models from an online repository or local directory.
'ultralytics/yolov5': Specifies the source repo of the YOLOv5 implementation, hosted by Ultralytics on GitHub.
'custom': Indicates that we are loading a model with custom-trained weights.
path='best.pt': Specifies the path to the trained YOLOv5 model weights file (best.pt). This file is generated after training your model on a custom dataset.
force_reload=True: Forces the script to reload the YOLOv5 model, even if it's already cached.
'''
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

# Define thresholds
BLINKING_THRESHOLD = 2.0  # Time (in seconds) before classifying as sleeping
FRAME_RATE = 30  # Assume video is running at 30 FPS

# Initialize tracking variables
blink_start_time = None
is_sleeping = False

# Open video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or replace with video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 inference
    results = model(frame)
    
    # Parse detection results
    detected_class = None
    for *box, conf, cls in results.xyxy[0]:  # Iterate through detections
        label = model.names[int(cls)]
        if label in ["blinking", "neutral", "yawning"]:
            detected_class = label
            break  # Process only the first detected face

    # Handle blinking detection logic
    if detected_class == "blinking":
        if blink_start_time is None:
            blink_start_time = time.time()  # Start tracking blink duration
        elif time.time() - blink_start_time > BLINKING_THRESHOLD:
            is_sleeping = True  # Eyes have been closed for too long
    else:
        blink_start_time = None  # Reset blink timer
        is_sleeping = False

    # Display status on the frame
    status_text = "Neutral"
    if detected_class == "yawning":
        status_text = "Yawning"
    elif detected_class == "blinking":
        status_text = "Blinking"
    if is_sleeping:
        status_text = "SLEEPING!"

    # Draw status text
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_sleeping else (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Driver Monitoring", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
