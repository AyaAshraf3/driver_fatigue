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
from PyQt5.QtCore import Qt
from thresholds import *  # Import thresholds for blink and yawn detection
from gaze_head_detection import GazeHeadDetection  # Import gaze and head pose detection functions


class DrowsinessDetector(QMainWindow):  # Defines DrowsinessDetector, inheriting from QMainWindow (PyQt5 GUI)
    def __init__(self):
        super().__init__()

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

        self.setWindowTitle("Driver Fatigue Detection")
        self.setGeometry(100, 100, 900, 600)
        self.setStyleSheet("background-color: white;")

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QHBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setStyleSheet("border: 2px solid black;")
        self.video_label.setFixedSize(640, 480)
        self.layout.addWidget(self.video_label)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("background-color: white; border: 1px solid black; padding: 10px;")
        self.layout.addWidget(self.info_label)


        # Load YOLO model
        self.detect_drowsiness = YOLO(r"D:\GRAD_PROJECT\driver_fatigue\models\best_ours.pt")

        self.cap = cv2.VideoCapture(0) # Capture video from webcam
        time.sleep(1.000)
        
        # Initialize Gaze & Head Detection Module
        self.gaze_head_detector = GazeHeadDetection()
        self.gaze_head_detector.start()  # Start gaze and head detection
        self.update_info()

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
            self.alert_text += "<p style='color: red; font-weight: bold;'>⚠️ Alert: Prolonged Microsleep Detected!</p>"
        if round(self.yawn_duration, 2) > yawning_threshold:
            self.alert_text += "<p style='color: orange; font-weight: bold;'>⚠️ Alert: Prolonged Yawn Detected!</p>"

        # **BASELINE NOT SET (SHOW CALIBRATION MESSAGE)**
        if baseline_flag == 0:
            info_text = (
                f"<div style='font-family: Arial, sans-serif; color: #333; padding: 10px; text-align: center;'>"
                f"<h2 style='color: #4CAF50;'>🚗 Drowsiness Detector 🚗</h2>"
                f"<hr style='border: 2px solid #4CAF50;'>"
                f"<h3 style='color: #FF9800;'>⚙️ Setting Baseline... Please Keep Your Head Straight ⚙️</h3>"
                f"<p style='color: #607D8B;'>We are calibrating normal head and gaze positions.</p>"
                f"<p style='color: #607D8B;'>Stay still for a few seconds.</p>"
                f"<hr style='border: 2px solid #4CAF50;'>"
                f"</div>"
            )

        # **BASELINE SET (SHOW NORMAL DETECTION INFO)**
        else:
            info_text = (
                f"<div style='font-family: Arial, sans-serif; color: #333; padding: 10px;'>"
                f"<h2 style='text-align: center; color: #4CAF50;'>🚗 Drowsiness Detector 🚗</h2>"
                f"<hr style='border: 2px solid #4CAF50;'>"

                # Alert Section
                f"<p style='color: red; font-weight: bold; text-align: center;'>{self.alert_text}</p>"

                # Blink & Yawn Statistics
                f"<h3 style='color: #2196F3;'>🔍 Detection Stats</h3>"
                f"<p><b>👀 Blinks:</b> <span style='color: #FF9800;'>{self.num_of_blinks}</span></p>"
                f"<p><b>😴 Microsleeps:</b> <span style='color: #E91E63;'>{round(self.microsleep_duration,2)} seconds</span></p>"
                f"<p><b>😮 Yawns:</b> <span style='color: #795548;'>{self.num_of_yawns}</span></p>"
                f"<p><b>⏳ Yawning Duration:</b> <span style='color: #9C27B0;'>{round(self.yawn_duration,2)} seconds</span></p>"
                f"<p><b>📊 Blinks per minute:</b> <span style='color: #009688;'>{self.blinks_per_minute} BPM</span></p>"
                f"<p><b>📊 Yawns per minute:</b> <span style='color: #009688;'>{self.yawns_per_minute} YPM</span></p>"

                f"<hr style='border: 2px solid #4CAF50;'>"

                # Gaze and Head Movement Section
                f"<h3 style='color: #2196F3;'>👁️ Gaze & Head Tracking</h3>"
                f"<p><b>🟡 Gaze Status:</b> <span style='color: {'red' if gaze_status == 'ABNORMAL GAZE' else 'green'};'>{gaze_status}</span></p>"
                f"<p><b>👀 Gaze Direction:</b> <span style='color: #673AB7;'>{gaze_direction}</span></p>"
                f"<p><b>🔴 Head Movement Status:</b> <span style='color: {'red' if head_status == 'ABNORMAL' else 'green'};'>{head_status}</span></p>"
                f"<p><b>📏 Pitch:</b> <span style='color: #607D8B;'>{pitch:.2f}</span></p>"
                f"<p><b>📏 Yaw:</b> <span style='color: #607D8B;'>{yaw:.2f}</span></p>"
                f"<p><b>📏 Roll:</b> <span style='color: #607D8B;'>{roll:.2f}</span></p>"

                f"<hr style='border: 2px solid #4CAF50;'>"

                # Distraction Counter
                f"<h3 style='color: #3F51B5;'>⚠️ Distraction Monitoring</h3>"
                f"<p><b>⚠️ Distraction Count:</b> <span style='color: #D32F2F; font-weight: bold;'>{distraction_counts}</span></p>"

                f"</div>"
            )

        # **Update the UI**
        self.info_label.setText(info_text)


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
        """Displays the processed frame in the PyQt5 GUI."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

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
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())
