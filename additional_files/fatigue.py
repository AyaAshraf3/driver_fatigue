from collections import deque
import cv2

# Parameters
BLINK_THRESHOLD = 30  # Number of consecutive frames to consider as fatigue
blink_history = deque(maxlen=BLINK_THRESHOLD)

def process_detections(detections, frame):
    global blink_history

    # Reset blink history if no face is detected
    if detections.empty:
        blink_history.clear()
        return

    # Get the most confident detection (assuming one face per frame)
    detection = detections.iloc[0]
    class_name = detection['name']
    confidence = detection['confidence']

    # Update blink history
    if class_name == 'blinking':
        blink_history.append(1)  # 1 for blinking
    else:
        blink_history.append(0)  # 0 for not blinking

    # Check for prolonged blinking
    if sum(blink_history) >= BLINK_THRESHOLD:
        print("Fatigue Alert: Prolonged Blinking Detected!")
        cv2.putText(frame, "Fatigue Alert: Prolonged Blinking!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check for yawning
    if class_name == 'yawning':
        print("Fatigue Alert: Yawning Detected!")
        cv2.putText(frame, "Fatigue Alert: Yawning!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)