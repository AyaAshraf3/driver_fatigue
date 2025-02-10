import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt , QTimer
from thresholds import *  # Import thresholds for blink and yawn detection
from gaze_head_detection import GazeHeadDetection  # Import gaze and head pose detection functions


import queue  # Used for thread-safe frame buffering
import threading  # Handles video capture and processing in parallel
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import sys
import tkinter as tk
from tkinter import Label, Frame, Canvas
from PIL import Image, ImageTk
from thresholds import *  # Import thresholds for blink and yawn detection
from gaze_head_detection import GazeHeadDetection  # Import gaze and head pose detection functions


class DrowsinessDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Driver Fatigue Detection")
        self.root.geometry("900x600")
        self.root.configure(bg="white")

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
        self.start_time = time.time()

        self.eyes_still_closed = False
        self.yawn_in_progress = False

        # Layout setup
        self.main_frame = Frame(self.root, bg="white")
        self.main_frame.pack(pady=10)

        # Video Frame
        self.video_canvas = Canvas(self.main_frame, width=640, height=480, bg="black")
        self.video_canvas.grid(row=0, column=0, padx=10)

        # Info Display
        self.info_label = Label(self.main_frame, text="Initializing...", bg="white", fg="black", font=("Arial", 12), justify="left")
        self.info_label.grid(row=0, column=1, padx=10, sticky="nw")

        # Load YOLO model
        self.detect_drowsiness = YOLO(r"D:\GRAD_PROJECT\driver_fatigue\models\best_ours.pt")

        # Start Webcam Capture
        self.cap = cv2.VideoCapture(0)  
        time.sleep(1.0)

        # Initialize Gaze & Head Detection Module
        self.gaze_head_detector = GazeHeadDetection()
        self.gaze_head_detector.start()

        # Threading Setup
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.blink_yawn_thread = threading.Thread(target=self.update_blink_yawn_rate)

        self.capture_thread.start()
        self.process_thread.start()
        self.blink_yawn_thread.start()

        # Start UI update loop
        self.update_info()

        

    def fatigue_detection(self,frame,blink_rate,yawning_rate, gaze_status, head_status):
        """Detects fatigue and distraction warnings based on multiple factors."""
        self.alert_text = ""
        if blink_rate > 35 or yawning_rate > 5:
            self.alert_text = "🚨 High Fatigue Risk!"
        elif blink_rate > 25 or yawning_rate > 3:
            self.alert_text = "⚠️ Possible Fatigue!"
            
        if gaze_status == "ABNORMAL GAZE":
            self.alert_text += " 🚨 Abnormal Gaze Detected!"
        
        if head_status == "ABNORMAL":
            self.alert_text += " 🚨 Unsafe Head Movement!"

    def update_info(self):
        with self.gaze_head_detector.lock:
            pitch = self.gaze_head_detector.pitch
            yaw = self.gaze_head_detector.yaw
            roll = self.gaze_head_detector.roll
            gaze_direction = self.gaze_head_detector.gaze_direction
            gaze_status = self.gaze_head_detector.gaze_status_gui
            baseline_flag = self.gaze_head_detector.baseline_flag
            distraction_counts = self.gaze_head_detector.distraction_counter
            head_status = "ABNORMAL" if self.gaze_head_detector.distraction_flag_head else "NORMAL"

        # Generate alert messages if necessary
        self.alert_text = ""
        if round(self.microsleep_duration, 2) > microsleep_threshold:
            self.alert_text += "⚠️ Alert: Prolonged Microsleep Detected!\n"
        if round(self.yawn_duration, 2) > yawning_threshold:
            self.alert_text += "⚠️ Alert: Prolonged Yawn Detected!\n"

        # BASELINE NOT SET (SHOW CALIBRATION MESSAGE)
        if baseline_flag == 0:
            info_text = (
                "🚗 Drowsiness Detector 🚗\n"
                "=========================\n"
                "⚙️ Setting Baseline... Please Keep Your Head Straight ⚙️\n"
                "We are calibrating normal head and gaze positions.\n"
                "Stay still for a few seconds.\n"
                "=========================\n"
            )
        # BASELINE SET (SHOW NORMAL DETECTION INFO)
        else:
            info_text = (
                "🚗 Drowsiness Detector 🚗\n"
                "=========================\n"
                f"{self.alert_text}\n"
                "🔍 Detection Stats:\n"
                f"👀 Blinks: {self.num_of_blinks}\n"
                f"😴 Microsleeps: {round(self.microsleep_duration,2)} sec\n"
                f"😮 Yawns: {self.num_of_yawns}\n"
                f"⏳ Yawning Duration: {round(self.yawn_duration,2)} sec\n"
                f"📊 Blinks per min: {self.blinks_per_minute} BPM\n"
                f"📊 Yawns per min: {self.yawns_per_minute} YPM\n"
                "=========================\n"
                "👁️ Gaze & Head Tracking\n"
                f"🟡 Gaze Status: {'⚠️' if gaze_status == 'ABNORMAL GAZE' else '✅'} {gaze_status}\n"
                f"👀 Gaze Direction: {gaze_direction}\n"
                f"🔴 Head Status: {'⚠️' if head_status == 'ABNORMAL' else '✅'} {head_status}\n"
                f"📏 Pitch: {pitch:.2f}\n"
                f"📏 Yaw: {yaw:.2f}\n"
                f"📏 Roll: {roll:.2f}\n"
                "=========================\n"
                "⚠️ Distraction Monitoring\n"
                f"⚠️ Distraction Count: {distraction_counts}\n"
            )

        # Update the Tkinter Label with the generated text
        self.info_label.config(text=info_text)

        # Schedule the next update
        self.root.after(500, self.update_info)  # Update every 500ms



    def predict(self, frame):
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
        """Processes each frame to detect eye state and yawning and updates head/gaze tracking."""
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                # Send frame to Gaze & Head Detector
                self.gaze_head_detector.update_frame(frame)
                  # Get Gaze & Head Movement Data
                gaze_status = self.gaze_head_detector.gaze_status_gui
                head_status = "ABNORMAL" if self.gaze_head_detector.distraction_flag_head else "NORMAL"
                self.update_info()
                self.display_frame(frame)
                
                try:
                    self.eyes_state = self.predict(frame)  # Predict on whole frame
                except Exception as e:
                    print(f"Error in realizing the prediciton: {e}") 
 
                # Handle eye blink detection
                if self.eyes_state == "Closed Eye":
                    if not self.eyes_still_closed:
                        self.eyes_still_closed = True
                        self.num_of_blinks += 1
                        self.current_blinks += 1
                    self.microsleep_duration += 45 / 1000
                else:
                    self.eyes_still_closed = False 
                    self.microsleep_duration = 0

                # Handle yawn detection
                if self.eyes_state == "Yawning":
                    if not self.yawn_in_progress:
                        self.yawn_in_progress = True
                        self.yawn_finished = False
                    self.yawn_duration += 45 / 1000
                    if yawning_threshold < self.yawn_duration and self.yawn_finished is False:
                        self.yawn_finished = True
                        self.num_of_yawns += 1
                        self.current_yawns += 1
                else:
                    if self.yawn_in_progress:
                        self.yawn_in_progress = False
                        self.yawn_finished = True
                        self.yawn_duration = 0
                

                self.update_info()
                # Perform fatigue detection
                self.fatigue_detection(frame, self.blinks_per_minute, self.yawns_per_minute, gaze_status, head_status)
                self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()

    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_image)
        img = img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_canvas.image = imgtk  # Keep reference

    def update_blink_yawn_rate(self):
        """Updates blink and yawn rates every minute."""
        while not self.stop_event.is_set():
            time.sleep(self.time_window)  # Wait for 1 minute
            self.blinks_per_minute = self.current_blinks
            self.yawns_per_minute = self.current_yawns
            self.current_blinks = 0
            self.current_yawns = 0
            print(f"Updated Rates - Blinks: {self.blinks_per_minute} per min, Yawns: {self.yawns_per_minute} per min")
    


    def play_alert_sound(self):
        """Plays an alert sound for fatigue detection."""
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        """Runs the alert sound in a separate thread."""
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()

    def show_alert_on_frame(self, frame, text="Alert!"):
        """Overlays alert text on the frame."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (0, 0, 255)
        line_type = 2
        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)

    def stop(self):
        """Stops all processing."""
        self.stop_event.set()
        self.gaze_head_detector.stop()

if __name__ == "__main__":
    import tkinter as tk

    root = tk.Tk()
    app = DrowsinessDetector(root)  # Pass Tkinter root window to your class
    root.mainloop()  # Start Tkinter main event loop
