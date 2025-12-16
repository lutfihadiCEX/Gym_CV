import cv2
import math

class BenchPressCounter:
    def __init__(self, calibrate=False):
        self.count = 0
        self.stage = None
        self.calibrate = calibrate
        self.min_angle = 180.0 
        self.max_angle = 0.0 
        self.prev_angle = None 
        self.smoothed_angle = None
        self.smoothing_factor = 0.1

    def get_elbow_angle(self, landmarks):
        """Calculate angle based ONLY on the Right Arm using YOLOv8 keypoints."""
        
        if landmarks is None or len(landmarks) < 17:
            return None

        
        def calculate_angle(a, b, c):
            
            v1 = [a[0] - b[0], a[1] - b[1]] # BA (Elbow to Shoulder)
            v2 = [c[0] - b[0], c[1] - b[1]] # BC (Elbow to Wrist)
            
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            mag_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag_v2 = math.sqrt(v2[0]**2 + v2[1]**2)

            cos = dot_product / (mag_v1 * mag_v2 + 1e-6)

            angle_rad = math.acos(min(1, max(-1, cos)))
            return math.degrees(angle_rad)
                
        try:
            rs = landmarks[6] 
            re = landmarks[8] 
            rw = landmarks[10] 
        except IndexError:
            
            return None

        
        
        if rs[2] > 0.00 and re[2] > 0.00 and rw[2] > 0.00:
            
            right_angle = calculate_angle(rs, re, rw)
        else:
            return None

        
        
        avg_angle = right_angle
        self.prev_angle = avg_angle
        return avg_angle

    def update(self, landmarks, image):
        if landmarks is None:
            return image, self.count

        raw_angle = self.get_elbow_angle(landmarks)
        if raw_angle is None:
            return image, self.count
        
        
        if self.smoothed_angle is None:
            self.smoothed_angle = raw_angle
        else:
            self.smoothed_angle = (self.smoothing_factor * raw_angle) + ((1 - self.smoothing_factor) * self.smoothed_angle)
            
        angle = self.smoothed_angle 

        if self.calibrate:
            self.min_angle = min(self.min_angle, raw_angle)
            self.max_angle = max(self.max_angle, raw_angle)
            cv2.putText(image, "CALIBRATION MODE", (image.shape[1]//2 - 200, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 5)
            cv2.putText(image, f"Current Angle: {int(angle)} deg", (50, 180),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 4)
            cv2.putText(image, f"Min Angle (bottom/bent): {int(self.min_angle)} deg", (50, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            cv2.putText(image, f"Max Angle (top/extended): {int(self.max_angle)} deg", (50, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
            cv2.putText(image, "Do 5 slow reps - pause at top/bottom", (50, image.shape[0]-100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            return image, self.count

        
        
        DOWN_ANGLE = 105 
        UP_ANGLE = 170 
        RESET_ANGLE = 150 

        
        if angle < RESET_ANGLE and self.stage == "up":
            self.stage = "descent" 

        
        if angle < DOWN_ANGLE:
            self.stage = "down" 

        
        if angle > UP_ANGLE and self.stage == "down":
            self.stage = "up" 
            self.count += 1
            cv2.putText(image, "REP COMPLETE!", (image.shape[1]//2 - 300, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 10)

        
        cv2.putText(image, f"Reps: {self.count}", (50, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 6)
        cv2.putText(image, f"Stage: {self.stage or 'waiting'}", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        
        cv2.putText(image, f"Smoothed Angle: {int(angle)} deg", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        
        cv2.putText(image, f"UP Threshold: {UP_ANGLE} deg", (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2) 
        cv2.putText(image, f"DOWN Threshold: {DOWN_ANGLE} deg", (50, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2) 
 
        return image, self.count