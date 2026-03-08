import cv2
import mediapipe as mp
import time

import config
from modules.camera.stream import CameraStream
from modules.vision.detector import YuNetDetector
from modules.vision.tracker import PanTiltTracker
from features.fatigue_warning.analyzer import analyze_fatigue
from modules.communication.mqtt_worker import MQTTWorker

def main():
    p_time = 0
    
    cam = CameraStream(source=0, width=config.WIDTH, height=config.HEIGHT, is_local=True)
    detector = YuNetDetector(model_path=config.MODEL_PATH, input_w=config.WIDTH, input_h=config.HEIGHT)
    tracker = PanTiltTracker(input_w=config.WIDTH, input_h=config.HEIGHT, edge_margin=60)
    
    worker = MQTTWorker()
    worker.start()
    
    mp_draw = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    spec_green = mp_draw.DrawingSpec(color=config.COLOR_GREEN, thickness=1, circle_radius=1)
    spec_blue = mp_draw.DrawingSpec(color=config.COLOR_BLUE, thickness=1, circle_radius=1)
    
    sleep_counter = 0
    yawn_counter = 0
    status = "Awake & Focused"
    status_color = config.COLOR_GREEN
    last_status_code = -1

    cv2.namedWindow("Robot Vision System", cv2.WINDOW_NORMAL)

    while True:
        success, frame = cam.read_frame()
        if not success: 
            break

        faces = detector.detect(frame)
        frame, pan_cmd, tilt_cmd = tracker.process_and_draw(frame, faces)
        
        h, w, _ = frame.shape
        avg_ear, mar = 0.0, 0.0

        if "CENTER" in pan_cmd and "CENTER" in tilt_cmd:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            
            if results.multi_face_landmarks:
                frame, avg_ear, mar, sleep_counter, yawn_counter, status, status_color, status_code = analyze_fatigue(
                    frame, results, w, h, sleep_counter, yawn_counter, 
                    mp_draw, mp_face_mesh, spec_green, spec_blue
                )
                
                if status_code != last_status_code:
                    if status_code != 0:
                        worker.publish("robot/status", {"code": status_code})
                    last_status_code = status_code
        else:
            sleep_counter = 0
            yawn_counter = 0
            status = "Adjusting Camera..."
            status_color = config.COLOR_YELLOW
                
        cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 2)
        cv2.putText(frame, f"STATUS: {status}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_GREEN, 2)
        
        cv2.imshow("Robot Vision System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    worker.stop()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()