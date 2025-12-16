from ultralytics import YOLO
import cv2

class PoseDetector:
    def __init__(self, model_path='yolov8n-pose.pt'):

        self.model = YOLO(model_path)
        self.right_arm_indices = [6, 8, 10] # Shoulder, Elbow, Wrist
        self.right_arm_ids = {6: 'RShoulder', 8: 'RElbow', 10: 'RWrist'}

    def process(self, image):
        # Run inference, conf=0.5 
        results = self.model(image, stream=False, verbose=False, conf=0.5)
        self.results = results
        
        # Keypoints for 1st detected human
        if results and len(results[0].keypoints.data) > 0:
            # Use data tensor -> keypoints
            import numpy as np
            keypoints_tensor = results[0].keypoints.data[0]             #  (17, 3) tensor
            
            xy = results[0].keypoints.xyn[0].cpu().numpy()              #  Normalized x, y (shape 17, 2)
            
            conf = results[0].keypoints.conf[0].cpu().numpy()           #  Confidence scores (shape 17)
            
            
            keypoints_xyc = np.hstack((xy, conf[:, None]))              #  (17, 3) array: [x, y, conf]
            
            return keypoints_xyc
            
        return None

    def draw(self, image):
        # Bounding boxes on human
        if self.results:
            annotated_frame = self.results[0].plot()
            return annotated_frame
        return image