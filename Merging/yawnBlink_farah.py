import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import sys
from thresholds import *  # Import thresholds for blink and yawn detection




num_of_blinks_gui=0
microsleep_duration_gui=0
num_of_yawns_gui=0
yawn_duration_gui=0
blinks_per_minute_gui=0
yawns_per_minute_gui=0


#flags for alert on gui
possibly_fatigued_alert=0
highly_fatigued_alert=0
possible_fatigue_alert=0


class DrowsinessDetector(): 
    def __init__(self):
        super().__init__()
        global num_of_blinks_gui
        global microsleep_duration_gui
        global num_of_yawns_gui
        global yawn_duration_gui
        global blinks_per_minute_gui
        global yawns_per_minute_gui


        #flags for alert on gui
        global possibly_fatigued_alert
        global highly_fatigued_alert
        global possible_fatigue_alert
        # Store current states
        self.yawn_state = ''
        self.eyes_state = ''
        self.alert_text = ''

        # Track statistics
        self.num_of_blinks = 0
        self.microsleep_duration = 0
        self.num_of_yawns = 0
        self.yawn_duration = 0

        # Track blinks/yawns per minute
        self.blinks_per_minute = 0
        self.yawns_per_minute = 0
        self.current_blinks = 0
        self.current_yawns = 0
        self.time_window = 60  # 1 minute window
        self.start_time = time.time()  # Track start time

        self.eyes_still_closed = False  # Track closed eye state
        self.yawn_in_progress = False # Track yawning state


        # Load YOLO model
        self.detect_drowsiness = YOLO(r"D:\GRAD_PROJECT\driver_fatigue\models\best_ours2.pt")

        self.cap = cv2.VideoCapture(0) # Capture video from webcam
        time.sleep(1.000)

        # Using Multi-Threading
        '''
        frame_queue → Stores frames for processing.
        capture_thread → Captures frames.
        process_thread → Processes frames.
        '''
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)  # Thread for tracking blinks/yawns per minute

        self.capture_thread.start()
        self.process_thread.start()
        self.blink_yawn_thread.start()  # Start the blink/yawn tracking thread
        


    def predict(self, frame):
        global num_of_blinks_gui
        global microsleep_duration_gui
        global num_of_yawns_gui
        global yawn_duration_gui
        global blinks_per_minute_gui
        global yawns_per_minute_gui
        """Sends the full frame for eye and yawn detection."""
        results = self.detect_drowsiness.predict(frame)
        boxes = results[0].boxes
        if len(boxes) == 0: #didn't detect anything (eyes)
            return "No Detection"

        confidences = boxes.conf.cpu().numpy() #take all confidences
        class_ids = boxes.cls.cpu().numpy() #take all classes ids in the yolo model
        max_confidence_index = np.argmax(confidences) #choose the max confidence and save its index
        class_id = int(class_ids[max_confidence_index]) #get the predict class by the index of the max confidence

        if class_id == 0: # and confidences[max_confidence_index] > eye_open_threshold
            return "Opened Eye"
        elif class_id == 1:
            return "Closed Eye"
        elif class_id == 2:
            return "Yawning"
        else:
            return "No Yawn"

    def capture_frames(self):
        """Continuously captures frames from webcam."""
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def process_frames(self):
        global num_of_blinks_gui
        global microsleep_duration_gui
        global num_of_yawns_gui
        global yawn_duration_gui
        global blinks_per_minute_gui
        global yawns_per_minute_gui
        """Processes each frame to detect eye state and yawning."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                try:
                    self.eyes_state = self.predict(frame)  # Predict on whole frame
                except Exception as e:
                    print(f"Error in realizing the prediciton: {e}") 
 
                # Handle eye blink detection
                if self.eyes_state == "Closed Eye":
                    if not self.eyes_still_closed:
                        self.eyes_still_closed = True
                        self.num_of_blinks += 1
                        num_of_blinks_gui=self.num_of_blinks
                        self.current_blinks += 1
                    self.microsleep_duration += 45 / 1000
                    microsleep_duration_gui = self.microsleep_duration
                else:
                    self.eyes_still_closed = False
                    self.microsleep_duration = 0
                    microsleep_duration_gui = self.microsleep_duration

                # Handle yawn detection
                if self.eyes_state == "Yawning":
                    if not self.yawn_in_progress:
                        self.yawn_in_progress = True
                        self.yawn_finished = False
                    self.yawn_duration += 45 / 1000
                    yawn_duration_gui=self.yawn_duration
                    if yawning_threshold < self.yawn_duration and self.yawn_finished is False:
                        self.yawn_finished = True
                        self.num_of_yawns += 1
                        num_of_yawns_gui = self.num_of_yawns 
                        self.current_yawns += 1
                else:
                    if self.yawn_in_progress:
                        self.yawn_in_progress = False
                        self.yawn_finished = True
                        self.yawn_duration = 0
                        yawn_duration_gui=self.yawn_duration
                

                #self.update_info()
                #self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()



    def update_blink_yawn_rate(self):
        global num_of_blinks_gui
        global microsleep_duration_gui
        global num_of_yawns_gui
        global yawn_duration_gui
        global blinks_per_minute_gui
        global yawns_per_minute_gui
        """Updates blink and yawn rates every minute."""
        while not self.stop_event.is_set():
            time.sleep(self.time_window)  # Wait for 1 minute
            self.blinks_per_minute = self.current_blinks
            blinks_per_minute_gui=self.blinks_per_minute
            self.yawns_per_minute = self.current_yawns
            yawns_per_minute_gui=self.yawns_per_minute
            self.current_blinks = 0
            self.current_yawns = 0
            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")
    

    def fatigue_detection(self,frame,blink_rate,yawning_rate,microsleep_duration):
        global possibly_fatigued_alert
        global highly_fatigued_alert
        global possible_fatigue_alert
        if microsleep_duration > microsleep_threshold:
            #self.play_sound_in_thread()
            #self.show_alert_on_frame(frame, "Alert! Possible fatigue!")
            possible_fatigue_alert=1
        if blink_rate > 35 or yawning_rate > 5:
            #self.show_alert_on_frame(frame, "Alert! Driver is Highly fatigued!")
            highly_fatigued_alert=1
        elif blink_rate > 25 or yawning_rate > 3:
            #self.show_alert_on_frame(frame, "Alert! Driver is possibly fatigued!")
            possibly_fatigued_alert=1


    def play_alert_sound(self):
        """Plays an alert sound for fatigue detection."""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        """Runs the alert sound in a separate thread."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()



