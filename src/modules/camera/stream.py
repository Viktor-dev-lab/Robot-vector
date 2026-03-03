import sys

import cv2

class CameraStream:
    def __init__(self, url, width=480, height=640):
        self.url = url
        self.width = width
        self.height = height
        print(f"Connecting to: {self.url}...")
        self.cap = cv2.VideoCapture(self.url)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not CameraStream.is_opened(self):
            print("Failed to connect to camera!")
            sys.exit(1)
        
    def is_opened(self):
        return self.cap.isOpened()
    
    def read_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None
        
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height))
        
        return True, frame
    
    def release(self):
        self.cap.release()