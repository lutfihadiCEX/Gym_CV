import cv2
import mediapipe as mp

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,                  # 1 = better accuracy (use 2 if too slow)
            smooth_landmarks=True,
            min_detection_confidence=0.4,        # Higher = stricter detection (helps avoid leg confusion)
            min_tracking_confidence=0.4
        )

    def process(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.pose.process(rgb)
        rgb.flags.writeable = True
        return self.results

    def draw(self, image):
        if self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image, self.results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=4, circle_radius=4),  # Yellow dots
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3)
            )
        return image