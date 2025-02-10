import mediapipe as mp
import numpy as np
import threading
import time
import cv2
from thresholds import *

class GazeHeadDetection(threading.Thread):
    def __init__(self):
        super().__init__()
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.gaze_direction="Center"
        self.gaze_status = "Normal"
        self.head_status = "Normal"
        self.running = True
        self.frame = None  # Frame to be processed
        self.lock = threading.Lock()  # Lock for thread-safe frame access
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Initialize timers and state variables
        self.start_time = time.time()  # Start timer to establish baseline for head movements
        self.baseline_pitch, self.baseline_yaw, self.baseline_roll = 0, 0, 0  # Initialize baseline angles for head movement
        self.baseline_data = []     # Store head movement angles for baseline calculation
        self.baseline_set = False   # Flag to indicate whether baseline is established
        self.baseline_flag = 0
        self.distraction_counter = 0
        #time_limit_counter = 100  #  minutes in seconds
        self.start_time_counter = time.time()  # Initialize start time
        self.no_blink_start_time = None  # Timer for no blink detection

        self.temp=0
        self.temg_g=0
        self.baseline_flag=0

        # Global variables for gaze and head movement detection 
        self.gaze_start_time = None        # Start time for abnormal gaze detection
        self.gaze_alert_triggered = False  # Flag to indicate abnormal gaze

        self.head_alert_start_time = None  # Start time for abnormal head movement detection
        self.head_alert_triggered = False  # Flag to indicate abnormal head movement
        # Flag to ensure abnormal gaze is counted only once
        self.gaze_flag = False 
        self.distraction_flag_head=0
        self.distraction_flag_gaze=0
        
        #GUI flags
        self.flag_gui= False
        self.gaze_gui='Center'
        self.gaze_status_gui= 0


        self.elapsed_time_counter=0
        self.time_limit_counter=5
        
    def calculate_angles(self, landmarks, frame_width, frame_height):
        # Compute angles (same logic as before)
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye_outer = landmarks[33]
        right_eye_outer = landmarks[263]
        forehead = landmarks[10]

        def normalized_to_pixel(normalized, width, height):
            return int(normalized.x * width), int(normalized.y * height)

        nose_tip = normalized_to_pixel(nose_tip, frame_width, frame_height)
        chin = normalized_to_pixel(chin, frame_width, frame_height)
        left_eye_outer = normalized_to_pixel(left_eye_outer, frame_width, frame_height)
        right_eye_outer = normalized_to_pixel(right_eye_outer, frame_width, frame_height)
        forehead = normalized_to_pixel(forehead, frame_width, frame_height)

        nose_3d = np.array([landmarks[1].x * frame_width, landmarks[1].y * frame_height, landmarks[1].z * frame_width])
        chin_3d = np.array([landmarks[152].x * frame_width, landmarks[152].y * frame_height, landmarks[152].z * frame_width])

        # Compute pitch using 3D coordinates
        pitch = self.calculate_pitch(nose_3d, chin_3d)

        delta_x_eye = right_eye_outer[0] - left_eye_outer[0]
        delta_y_eye = right_eye_outer[1] - left_eye_outer[1]
        yaw = np.arctan2(delta_y_eye, delta_x_eye) * (180 / np.pi)

        delta_x_forehead = forehead[0] - chin[0]
        delta_y_forehead = forehead[1] - chin[1]
        roll = np.arctan2(delta_y_forehead, delta_x_forehead) * (180 / np.pi)
        return pitch, yaw, roll 


    def calculate_pitch(self,nose, chin):
        """Compute the pitch angle using nose and chin landmarks."""
        # Vector from nose to chin (3D)
        #positive value means the chin is lower than the nose (head tilted downward "forward")
        #negative value means the chin is above the nose (head tilted upward "backword")
        vector = np.array([chin[0] - nose[0], chin[1] - nose[1], chin[2] - nose[2]])

        # Compute pitch using atan2 (Preserves sign for up/down movement)
        #projection of the vector in the X-Z plane "np.linalg.norm([vector[0], vector[2]])"
        pitch_angle = np.degrees(np.arctan2(vector[1], np.linalg.norm([vector[0], vector[2]])))

        # Adjust for backward movement
        if chin[2] > nose[2]:  # If chin is deeper into the screen than the nose
            pitch_angle *= -1   # Invert pitch to reflect backward movement correctly

        return pitch_angle

    
    # Function to detect abnormal center gaze "without blinking"
    def process_blink_and_gaze(self, gaze, left_ear , left_iris_position_y):
        if gaze == "Center":
            if self.no_blink_start_time is None:
                self.no_blink_start_time = time.time()
            
            else:
                elapsed_time = time.time() - self.no_blink_start_time
                if elapsed_time >= NO_BLINK_GAZE_DURATION:
                    if not self.gaze_alert_triggered:
                        self.gaze_alert_triggered = True  # Consider prolonged center gaze without blinking as abnormal
                        gaze = "Center Gazed"        # Set gaze to "Center Gazed" to trigger alert
                    
        else:
            self.no_blink_start_time = None    # Reset timer if gaze changes
    
        if left_iris_position_y < -0.3 and left_ear < EAR_THRESHOLD:
            self.no_blink_start_time = None    # Reset blink timer if blink is detected
            self.gaze_alert_triggered = False  # Reset abnormal gaze trigger if blinking occurs
            gaze = "Down"
        return gaze
    
    def reset_distraction_flag(self):
        # Reset distraction flags
        self.distraction_flag_head = 0  
        self.distraction_flag_gaze = 0  

        # Reset distraction counter after buzzer stops
        self.distraction_counter = 0  
        self.start_time_counter = time.time()  

        # Run action_after_buzzer() after 1 second
        #root.after(1000, action_after_buzzer)
    
    def compute_ear(self,landmarks, eye_indices):
        # Vertical distances
        #Lower and upper eyelid of the left eye
        vertical1 = np.linalg.norm(
            np.array([landmarks[159].x, landmarks[159].y]) - 
            np.array([landmarks[145].x, landmarks[145].y])
        )
        #Lower and upper eyelid of the right eye
        vertical2 = np.linalg.norm(
            np.array([landmarks[158].x, landmarks[158].y]) - 
            np.array([landmarks[144].x, landmarks[144].y])
        )
        # Horizontal distance
        #Outer and inner corners of the left eye
        horizontal = np.linalg.norm(
            np.array([landmarks[33].x, landmarks[33].y]) - 
            np.array([landmarks[133].x, landmarks[133].y])
        )
        # Compute EAR
        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return ear
    
   
    def run(self):
        
        # Initialize distraction time counter
        self.start_time_counter = time.time()  # Start tracking distractions
        time_limit_counter = 180  # 3-minute threshold
        left_eye_indices = [33, 133, 160, 159, 158, 144, 145, 153]
        left_iris_indices = [468, 469, 470, 471]
        while self.running:
            if self.frame is not None:
                 # Calculate elapsed time for distraction tracking
                elapsed_time_counter = time.time() - self.start_time_counter
                with self.lock:
                    frame = self.frame.copy()  # Safely copy the frame for processing

                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)

                # -------------------------------------------- Head Movement Detection --------------------------------------------
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        if not self.baseline_set:
                            self.pitch, self.yaw, self.roll =self.calculate_angles(face_landmarks.landmark, w, h)
                            elapsed_time = time.time() - self.start_time 
                            self.baseline_data.append((self.pitch, self.yaw, self.roll))  # Store collected angles
                            # ✅ Once baseline is collected for the threshold duration
                            if elapsed_time >= threshold_time:
                                # Compute baseline averages
                                self.baseline_pitch, self.baseline_yaw, self.baseline_roll = np.mean(self.baseline_data, axis=0)
                                self.baseline_set = True  # Mark baseline as set
                                self.baseline_flag=1
                                
                        
                        else:
                            # ✅ Step 2: Calculate head angles
                            self.flag_gui = 1  # Enable GUI updates
                            self.pitch, self.yaw, self.roll = self.calculate_angles(face_landmarks.landmark, w, h) 
                            
                            # Detect abnormal movements
                            head_alerts = []
                            if abs(self.pitch - self.baseline_pitch) > PITCH_THRESHOLD or self.pitch > 73:
                                head_alerts = self.check_abnormal_angles(self.pitch, self.yaw, self.roll, 'pitch')
                            if abs(self.yaw - self.baseline_yaw) > YAW_THRESHOLD:
                                head_alerts = self.check_abnormal_angles(self.pitch, self.yaw, self.roll, 'yaw')
                            if abs(self.roll - self.baseline_roll) > ROLL_THRESHOLD:
                                head_alerts = self.check_abnormal_angles(self.pitch, self.yaw, self.roll, 'roll')   
                            
                            # ✅ Step 3: Trigger alerts based on detected movements
                            if head_alerts:
                                if self.head_alert_start_time is None:  # Start timer
                                    self.head_alert_start_time = time.time()
                                elif time.time() - self.head_alert_start_time > head_abnormal_duration and not self.head_alert_triggered:
                                    self.head_alert_triggered = True
                                    self.distraction_counter += 1  # Increase distraction counter
                            else:
                                self.head_alert_start_time = None  # Reset timer
                                self.head_alert_triggered = False
        # -------------------------------------------- Gaze Detection --------------------------------------------
                           
                        def get_center(landmarks, indices):
                            points = np.array([[landmarks[i].x, landmarks[i].y] for i in indices])
                            return np.mean(points, axis=0)

                        left_eye_center = get_center(face_landmarks.landmark, left_eye_indices)
                        left_iris_center = get_center(face_landmarks.landmark, left_iris_indices)

                        left_eye_width = np.linalg.norm(
                            np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y]) - 
                            np.array([face_landmarks.landmark[133].x, face_landmarks.landmark[133].y])
                        )
                        left_iris_position_x = (left_iris_center[0] - left_eye_center[0]) / left_eye_width

                        left_eye_height = np.linalg.norm(
                            np.array([face_landmarks.landmark[159].x, face_landmarks.landmark[159].y]) - 
                            np.array([face_landmarks.landmark[145].x, face_landmarks.landmark[145].y])
                        )
                        left_iris_position_y = (left_iris_center[1] - left_eye_center[1]) / left_eye_height

                        # ✅ Step 1: Detect Gaze Direction
                        if left_iris_position_x < -0.1:
                            self.gaze_direction = "Right"
                        elif left_iris_position_x > 0.1:
                            self.gaze_direction = "Left"
                        else:
                            self.gaze_direction = self.process_blink_and_gaze("Center", 
                                                                           self.compute_ear(face_landmarks.landmark, left_eye_indices), 
                                                                           left_iris_position_y)
                    
                        
                        # ✅ Step 2: Detect Abnormal Gaze
                        if self.gaze_direction in ["Left", "Right", "Down", "Center Gazed"]:
                            if self.gaze_start_time is None:
                                self.gaze_start_time = time.time()
                            elif time.time() - self.gaze_start_time > gaze_abnormal_duration and not self.gaze_alert_triggered:
                                self.gaze_alert_triggered = True
                                if not self.gaze_flag:
                                    self.distraction_counter += 1
                                    self.gaze_flag = True
                        else:
                            self.gaze_start_time = None
                            self.gaze_alert_triggered = False

                            if self.gaze_direction == "Center":
                                self.gaze_flag=False
                                
                        # ✅ Step 3: Update Gaze Warnings
                        if self.gaze_alert_triggered:
                            self.gaze_status_gui = "ABNORMAL GAZE"
                            self.distraction_flag_gaze = 1
                        else:
                            self.gaze_status_gui = "NORMAL"
                            self.distraction_flag_gaze = 0
                            
                        # Store Head Movement Status
                        if abs(self.pitch) > 30 or abs(self.yaw) > 30 or abs(self.roll) > 20:
                            self.head_status = "ABNORMAL"
                            self.distraction_counter += 1
                        else:
                            self.head_status = "NORMAL"
    # -------------------------------------------- Distraction Handling --------------------------------------------


                        # ✅ If distraction threshold is reached, trigger HIGH RISK alert
                        if self.distraction_counter >= DISTRACTION_THRESHOLD and elapsed_time_counter < time_limit_counter:
                            self.temp = 1
                            self.temp_g = 1
                            self.distraction_flag_head = 2
                            self.distraction_flag_gaze = 2

                            # Activate buzzer alert
                            #buzzer_alert()
                            
                        elif self.elapsed_time_counter >= self.time_limit_counter:
                            # ✅ Reset counter every 3 minutes
                            print("⏳ 3 minutes passed. Resetting counter.")
                            self.distraction_counter = 0
                            self.start_time_counter = time.time()
                    
                time.sleep(0.05)  # Slight delay to avoid high CPU usage
                
    # Function to check if head movement angles exceed thresholds
    def check_abnormal_angles(self,pitch, yaw, roll, movement_type):
        alerts = []                     
        if movement_type == 'pitch':    
            alerts.append("Abnormal Pitch")
        if movement_type == 'yaw' :     
            alerts.append("Abnormal Yaw")
        elif movement_type == 'roll':  
            alerts.append("Abnormal Roll")
        return alerts  

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame  # Update frame safely

    def stop(self):
        self.running = False
