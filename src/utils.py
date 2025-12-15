import cv2
import time
import winsound

class FPS:

    def __init__(self):
        self.p_time = 0

    def update(self, img):
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if self.p_time != 0 else 0
        self.p_time = c_time
        cv2.putText(img, f"FPS: {int(fps)}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
def beep():
    winsound.Beep(1000, 200)  