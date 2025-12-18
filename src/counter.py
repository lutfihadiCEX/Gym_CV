import cv2
import math
import numpy as np
import time

class BenchPressCounter:
    def __init__(self, calibrate=False):
        self.count = 0
        self.stage = "up" 
        self.calibrate = calibrate
        self.min_angle = 180.0 
        self.max_angle = 0.0 
        self.smoothed_angle = None
        self.smoothing_factor = 0.2             # Modify for noise smoothing factor
        self.last_count_time = 0
        self.cooldown_period = 5.0              # Preventing accidental reps counted between lockouts
        self.path_history = []      
        self.max_path_points = 50               # Tail length


    def get_elbow_angle(self, landmarks):
        """Forces angle calculation by picking the best available arm."""
        try:
            # Convert input to array
            lms = np.array(landmarks)
            if lms.shape[0] < 11: return None

            def calc(a, b, c):
                # Std floats
                v1 = np.array([float(a[0]) - float(b[0]), float(a[1]) - float(b[1])])
                v2 = np.array([float(c[0]) - float(b[0]), float(c[1]) - float(b[1])])
                
                # Vector math 
                unit_v1 = v1 / (np.linalg.norm(v1) + 1e-6)
                unit_v2 = v2 / (np.linalg.norm(v2) + 1e-6)
                dot_product = np.dot(unit_v1, unit_v2)
                angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
                return np.degrees(angle)

            # Check Right Side (6, 8, 10) & Left Side (5, 7, 9)
            r_angle = calc(lms[6], lms[8], lms[10])
            l_angle = calc(lms[5], lms[7], lms[9])

            # Which arm is more extended
            # Bypass conf checks
            return max(r_angle, l_angle)

        except Exception as e:
            return None

    def update(self, landmarks, image):
        if landmarks is None:
            return image, self.count

        raw_angle = self.get_elbow_angle(landmarks)
        if raw_angle is None:
            return image, self.count
        
        # Smoothing
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = (self.smoothing_factor * raw_angle) + ((1 - self.smoothing_factor) * self.smoothed_angle)
            
        angle = self.smoothed_angle 

        if self.calibrate:
            self.min_angle = min(self.min_angle, raw_angle)
            self.max_angle = max(self.max_angle, raw_angle)
            cv2.putText(image, "CALIBRATION MODE", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(image, f"Angle: {int(angle)}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            cv2.putText(image, f"Min: {int(self.min_angle)} | Max: {int(self.max_angle)}", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return image, self.count

        # Logic Thresholds Angles
        DOWN_THR = 95 
        UP_THR = 165 
        RESET_THR = 130

        if angle < RESET_THR and self.stage == "up":
            self.stage = "descent" 

        if angle < DOWN_THR:
            self.stage = "down" 

        current_time = time.time()
        if angle > UP_THR and self.stage == "down":
            if (current_time - self.last_count_time) > self.cooldown_period:    
                self.stage = "up" 
                self.count += 1
                self.last_count_time = current_time

                cv2.putText(image, "LOCKOUT!", (image.shape[1]//2 - 100, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                
        if landmarks is not None:
            h, w, _ = image.shape
            wrist_x = int(landmarks[10][0] * w)
            wrist_y = int(landmarks[10][1] * h)
            current_pos = (wrist_x, wrist_y)

        if landmarks[10][2] > 0.5:
            self.path_history.append(current_pos)

        if len(self.path_history) > self.max_path_points:
            self.path_history.pop(0)

        for i in range(1, len(self.path_history)):
            thickness = int(np.linspace(1, 5, self.max_path_points)[i])
            cv2.line(image, self.path_history[i - 1], self.path_history[i], (0, 255, 0), thickness)

        if self.stage == "up" and (time.time() - self.last_count_time) < 0.1:
            self.path_history = []


        # UI
        cv2.putText(image, f"Reps: {self.count}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 4)
        cv2.putText(image, f"Stage: {self.stage}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
        cv2.putText(image, f"Angle: {int(angle)}", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        return image, self.count