import sys
import cv2

class CameraStream:
    # Using IP Camera source=config.CAMERA_URL, is_local=False
    def __init__(self, source=0, width=480, height=640, is_local=True):
        self.source = source
        self.width = width
        self.height = height
        self.is_local = is_local
        
        print(f"Connecting to camera: {self.source}...")
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        if not self.is_opened():
            print("Failed to connect to camera!")
            sys.exit(1)
            
    def is_opened(self):
        return self.cap.isOpened()
    
    def read_frame(self):
        success, frame = self.cap.read()
        if not success:
            return False, None
        
        if not self.is_local:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (self.width, self.height))
        
        return True, frame
    
    def release(self):
        self.cap.release()