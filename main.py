import cv2
import argparse
from src.detector import PoseDetector
from src.counter import BenchPressCounter
from src.utils import FPS, beep

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=0,
                        help="Path to video file or 0 for webcam")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run calibration to find best thresholds")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video if args.video != "0" else 0)
    detector = PoseDetector()
    counter = BenchPressCounter(calibrate=args.calibrate)
    fps_counter = FPS()

    print("Press 'q' to quit | Working on:", "Webcam" if args.video == 0 else args.video)

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Video ended or cannot read frame")
            break

        img = cv2.flip(img, 1) if args.video == "0" else img
        
        
        keypoints = detector.process(img) 
        img = detector.draw(img) 

        
        img, count = counter.update(keypoints, img)
        

        if counter.stage == "up" and count > 0 and count == counter.count: 
            beep()

        fps_counter.update(img)
        cv2.imshow("Gym Rep Counter - Bench Press", img)

        if cv2.waitKey(1) == ord('q'):
            break

    print(f"Final Rep Count: {counter.count}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()