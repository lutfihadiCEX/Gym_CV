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

    def get_elbow_angle(self, landmarks):
        """Calculate average elbow angle with visibility check + smoothing"""
        if not landmarks:
            return None

        def calculate_angle(a, b, c):
            # Vector AB and BC
            ab = [b.x - a.x, b.y - a.y]
            bc = [c.x - b.x, c.y - b.y]
            # Cosine angle
            cos = (ab[0]*bc[0] + ab[1]*bc[1]) / (math.sqrt(ab[0]**2 + ab[1]**2) * math.sqrt(bc[0]**2 + bc[1]**2) + 1e-6)
            return math.degrees(math.acos(min(1, max(-1, cos))))

        # Left arm
        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]
        if left_shoulder.visibility > 0.4 and left_elbow.visibility > 0.4 and left_wrist.visibility > 0.4:
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        else:
            left_angle = None

        # Right arm
        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]
        if right_shoulder.visibility > 0.4 and right_elbow.visibility > 0.4 and right_wrist.visibility > 0.4:
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        else:
            right_angle = None

        # Average 
        angles = [a for a in [left_angle, right_angle] if a is not None]
        if not angles:
            return self.prev_angle  

        avg_angle = sum(angles) / len(angles)
        self.prev_angle = avg_angle  
        return avg_angle

    def update(self, landmarks, image):
        if not landmarks:
            return image, self.count

        angle = self.get_elbow_angle(landmarks)
        if angle is None:
            cv2.putText(image, "No detection - adjust camera/light", (50, image.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return image, self.count

        if self.calibrate:
            self.min_angle = min(self.min_angle, angle)
            self.max_angle = max(self.max_angle, angle)
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
            self.draw_debug_boxes(landmarks, image)
            return image, self.count

        
        DOWN_ANGLE = 100  # Fully bent 
        UP_ANGLE = 160    # Lockout

        if angle < DOWN_ANGLE:
            self.stage = "down"
        if angle > UP_ANGLE and self.stage == "down":
            self.stage = "up"
            self.count += 1
            cv2.putText(image, "REP COMPLETE!", (image.shape[1]//2 - 300, 150),
                        cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0), 10)

        # Display
        cv2.putText(image, f"Reps: {self.count}", (50, 100),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 255, 255), 6)
        cv2.putText(image, f"Stage: {self.stage or 'waiting'}", (50, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 4)
        cv2.putText(image, f"Current Angle: {int(angle)} deg", (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        self.draw_debug_boxes(landmarks, image)
        return image, self.count

    def draw_debug_boxes(self, landmarks, image):
        """Yellow boxes around shoulder elbow-wrist (green good, red low confidence)"""
        h, w = image.shape[:2]
        points = [11, 13, 15, 12, 14, 16]  # Left + right: shoulder, elbow, wrist
        for idx in points:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            color = (0, 255, 0) if lm.visibility > 0.4 else (0, 0, 255)  
            cv2.rectangle(image, (x-25, y-25), (x+25, y+25), color, 3)
            cv2.putText(image, f"{lm.visibility:.2f}", (x-20, y-30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  