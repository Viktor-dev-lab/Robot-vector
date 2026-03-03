import cv2

class YuNetDetector:
    def __init__(self, model_path, input_w=480, input_h=640):
        try:
            self.detector = cv2.FaceDetectorYN.create(
                model=model_path,
                config="",
                input_size=(input_w, input_h),
                score_threshold=0.8,
                nms_threshold=0.3,
                top_k=5000
            )
        except Exception as e:
            raise Exception(f"Error loading YuNet model: {e}")

    def detect(self, frame):
        retval, faces = self.detector.detect(frame)
        return faces