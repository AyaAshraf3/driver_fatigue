import queue  #Used for thread-safe frame buffering.
import threading  # Handles video capture and processing in parallel.
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
from thresholds import *



class DrowsinessDetector(QMainWindow): #Defines DrowsinessDetector, inheriting from QMainWindow (PyQt5 GUI)
    def __init__(self):
        super().__init__()

        # Store current states.
        self.yawn_state = ''
        self.eyes_state =''
        self.left_eye_state = ''
        self.right_eye_state = ''
        self.alert_text = ''


        #Track statistics.
        self.num_of_blinks = 0
        self.microsleep_duration = 0
        self.num_of_yawns = 0
        self.yawn_duration = 0 

        self.left_eye_still_closed = False  
        self.right_eye_still_closed = False
        self.yawn_in_progress = False  
        
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        self.setWindowTitle("Driver fatigue Detection")
        self.setGeometry(100, 100, 800, 600)
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

        self.update_info()
        
        self.detectdrowsiness = YOLO(r"D:\grad project\driver_fatigue\models\best_ours.pt")

        self.cap = cv2.VideoCapture(0)
        time.sleep(1.000)

        #Using Multi-Threading
        '''
        frame_queue → Stores frames for processing.
        capture_thread → Captures frames.
        process_thread → Processes frames.
        '''
        self.frame_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()

        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)

        self.capture_thread.start()
        self.process_thread.start()
        
    def update_info(self):
        '''
        if round(self.yawn_duration, 2) > yawning_threshold:
            # self.play_sound_in_thread()
            self.alert_text = "<p style='color: orange; font-weight: bold;'>⚠️ Alert: Prolonged Yawn Detected!</p>"
        '''

        if round(self.microsleep_duration, 2) > microsleep_threshold:
            #self.play_sound_in_thread()
            self.alert_text = "<p style='color: red; font-weight: bold;'>⚠️ Alert: Prolonged Microsleep Detected!</p>"


        info_text = (
            f"<div style='font-family: Arial, sans-serif; color: #333;'>"
            f"<h2 style='text-align: center; color: #4CAF50;'>Drowsiness Detector</h2>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"{self.alert_text}"  # Display alert if it exists
            f"<p><b> Blinks:</b> {self.num_of_blinks}</p>"
            f"<p><b> Microsleeps:</b> {round(self.microsleep_duration,2)} seconds</p>"
            f"<p><b> Yawns:</b> {self.num_of_yawns}</p>"
            f"<p><b> Yawning Duration:</b> {round(self.yawn_duration,2)} seconds</p>"
            f"<hr style='border: 1px solid #4CAF50;'>"
            f"</div>"
        )
        self.info_label.setText(info_text)


    def predict(self, frame, state):
        results = self.detectdrowsiness.predict(frame)
        boxes = results[0].boxes
        if len(boxes) == 0: #didn't detect anything (eyes)
            return state

        confidences = boxes.conf.cpu().numpy()  #take all confidences
        class_ids = boxes.cls.cpu().numpy()  #take all classes ids in the yolo model
        max_confidence_index = np.argmax(confidences) #choose the max confidence and save its index
        class_id = int(class_ids[max_confidence_index]) #get the predict class by the index of the max confidence

        if class_id == 0 :
            state = "Close Eye"
        elif class_id == 1 and confidences[max_confidence_index] > eye_open_threshold:
            state = "Open Eye"
        elif class_id == 2 :
            state = "Yawn" 
        else:
            state = "No Yawn"
                            
        return state

                            

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.qsize() < 2:
                    self.frame_queue.put(frame)
            else:
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)        
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #OpenCV uses BGR, but MediaPipe requires RGB, so conversion is needed.
                results = self.face_mesh.process(image_rgb)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        ih, iw, _ = frame.shape
                        points = []

                        for point_id in self.points_ids:
                            lm = face_landmarks.landmark[point_id]
                            x, y = int(lm.x * iw), int(lm.y * ih)
                            points.append((x, y))

                        if len(points) != 0:
                            x1, y1 = points[0]  # Upper mouth
                            x2, _ = points[1]  # Lower mouth
                            _, y3 = points[2]  # Chin

                            x4, y4 = points[3]  # Left eye corner
                            x5, y5 = points[4]  # Right eye corner

                            x6, y6 = points[5]  # Left eye upper
                            x7, y7 = points[6]  # Left eye lower

                            #Ensures correct ordering of eye coordinates.
                            x6, x7 = min(x6, x7), max(x6, x7)
                            y6, y7 = min(y6, y7), max(y6, y7)

                            mouth_roi = frame[y1:y3, x1:x2]
                            right_eye_roi = frame[y4:y5, x4:x5]
                            left_eye_roi = frame[y6:y7, x6:x7]

                            try:
                                self.left_eye_state = self.predict(left_eye_roi, self.left_eye_state)
                                self.right_eye_state = self.predict(right_eye_roi, self.right_eye_state)
                                self.yawn_state = self.predict(mouth_roi,self.yawn_state)

                            except Exception as e:
                                print(f"Error in realizing the prediciton: {e}")

                            
                            if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
                                if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                    self.left_eye_still_closed, self.right_eye_still_closed = True , True
                                    self.num_of_blinks += 1 
                                self.microsleep_duration += 45 / 1000
                            else:
                                if self.left_eye_still_closed and self.right_eye_still_closed :
                                    self.left_eye_still_closed, self.right_eye_still_closed = False , False
                                self.microsleep_duration = 0

                            if self.yawn_state == "Yawn":
                                if not self.yawn_in_progress:
                                    self.yawn_in_progress = True
                                    self.yawn_finished = False
                                self.yawn_duration += 45 / 1000
                                if yawning_threshold < self.yawn_duration and self.yawn_finished is False:
                                    self.yawn_finished = True
                                    self.num_of_yawns += 1  
                            else:
                                if self.yawn_in_progress:
                                    self.yawn_in_progress = False
                                    self.yawn_finished = True
                                    self.yawn_duration = 0

                            self.update_info()
                            self.display_frame(frame)

            except queue.Empty:
                continue

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
    
    def fatigue_detection(self,frame,blink_rate,yawning_rate,microsleep_duration):
        if microsleep_duration > microsleep_threshold:
            #self.play_sound_in_thread()
            self.show_alert_on_frame(frame, "Alert! Possible fatigue!")
        if blink_rate > 35 or yawning_rate > 5:
            self.show_alert_on_frame(frame, "Alert! Driver is Highly fatigued!")
        elif blink_rate > 25 or yawning_rate > 3:
            self.show_alert_on_frame(frame, "Alert! Driver is possibly fatigued!")


    def display_frame(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(p))

    def play_alert_sound(self):
            frequency = 1000 
            duration = 500  
            winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()
        
    def show_alert_on_frame(self, frame, text="Alert!"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = (50, 50)
        font_scale = 1
        font_color = (0, 0, 255) 
        line_type = 2

        cv2.putText(frame, text, position, font, font_scale, font_color, line_type)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DrowsinessDetector()
    window.show()
    sys.exit(app.exec_())