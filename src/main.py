import cv2
import math
import mediapipe as mp
import time

# Import các module của bạn
from modules.camera.stream import CameraStream
from modules.vision.detector import YuNetDetector
from modules.vision.tracker import PanTiltTracker

# ==========================================
# 1. CÁC HÀM TÍNH TOÁN TOÁN HỌC
# ==========================================
def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(eye_indices, landmarks, w, h):
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    v1 = calculate_distance(eye_points[1], eye_points[5])
    v2 = calculate_distance(eye_points[2], eye_points[4])
    h_dist = calculate_distance(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

def calculate_mar(mouth_indices, landmarks, w, h):
    p_top = (int(landmarks[mouth_indices[0]].x * w), int(landmarks[mouth_indices[0]].y * h))
    p_bottom = (int(landmarks[mouth_indices[1]].x * w), int(landmarks[mouth_indices[1]].y * h))
    p_left = (int(landmarks[mouth_indices[2]].x * w), int(landmarks[mouth_indices[2]].y * h))
    p_right = (int(landmarks[mouth_indices[3]].x * w), int(landmarks[mouth_indices[3]].y * h))
    v_dist = calculate_distance(p_top, p_bottom)
    h_dist = calculate_distance(p_left, p_right)
    return v_dist / h_dist if h_dist != 0 else 0

# ==========================================
# 2. HÀM CHÍNH
# ==========================================
def main():
    MODEL_PATH = "model/face_detection_yunet_2023mar.onnx"
    CAMERA_URL = "http://admin:123@192.168.2.78:8081/video"
    WIDTH, HEIGHT = 480, 640
    pTime = 0

    # Khởi tạo các module Camera & Tracking
    cam = CameraStream(url=CAMERA_URL, width=WIDTH, height=HEIGHT)
    detector = YuNetDetector(model_path=MODEL_PATH, input_w=WIDTH, input_h=HEIGHT)
    tracker = PanTiltTracker(input_w=WIDTH, input_h=HEIGHT, edge_margin=60)
    
    # Khởi tạo MediaPipe
    mpDraw = mp.solutions.drawing_utils
    mpFaceMesh = mp.solutions.face_mesh
    faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Khởi tạo bút vẽ (Tối ưu: Đưa ra ngoài vòng lặp)
    spec_green = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
    spec_blue = mpDraw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    # Mốc tính toán Landmarks
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    MOUTH_INDICES = [13, 14, 78, 308] 

    # Cấu hình Ngưỡng (Thresholds)
    EAR_THRESHOLD = 0.26  
    MAR_THRESHOLD = 0.4   
    FRAMES_TO_SLEEP = 15  
    FRAMES_TO_YAWN = 12   

    sleep_counter = 0
    yawn_counter = 0
    status = "Awake & Focused"
    status_color = (0, 255, 0) 

    cv2.namedWindow("Robot Vision System", cv2.WINDOW_NORMAL)
    print("Kết nối thành công! Nhấn 'q' để tắt.")

    # ==========================================
    # 3. VÒNG LẶP PIPELINE
    # ==========================================
    while True:
        success, frame = cam.read_frame()
        if not success:
            print("Mất kết nối camera.")
            break

        # 1. AI phát hiện khuôn mặt & Tracking (Luôn chạy để bám mục tiêu)
        faces = detector.detect(frame)
        frame, pan_cmd, tilt_cmd = tracker.process_and_draw(frame, faces)
        
        h, w, c = frame.shape
        avg_ear = 0.0
        mar = 0.0

        # 2. LOGIC TỐI ƯU: Chỉ kiểm tra buồn ngủ khi mặt ở vùng an toàn (CENTER)
        if "CENTER" in pan_cmd and "CENTER" in tilt_cmd:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = faceMesh.process(frameRGB)
            
            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    # Vẽ đồ họa màu nổi bật
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, spec_green, spec_green)
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LEFT_EYE, spec_blue, spec_blue)
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_RIGHT_EYE, spec_blue, spec_blue)
                    mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LIPS, spec_blue, spec_blue)

                    # Tính toán EAR & MAR
                    landmarks = faceLms.landmark
                    left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks, w, h)
                    right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks, w, h)
                    avg_ear = (left_ear + right_ear) / 2.0
                    mar = calculate_mar(MOUTH_INDICES, landmarks, w, h)
                    
                    # Kiểm tra Ngáp
                    if mar > MAR_THRESHOLD:
                        yawn_counter += 1
                        if yawn_counter >= FRAMES_TO_YAWN:
                            status = "Yawning (Fatigue!)"
                            status_color = (0, 165, 255) 
                    else:
                        yawn_counter = 0
                    
                    # Kiểm tra Buồn ngủ
                    if avg_ear < EAR_THRESHOLD:
                        sleep_counter += 1
                        if sleep_counter >= FRAMES_TO_SLEEP:
                            status = "Sleeping (Drowsy!)"
                            status_color = (0, 0, 255) 
                    else:
                        sleep_counter = 0
                        
                    # Trạng thái bình thường
                    if sleep_counter == 0 and yawn_counter == 0:
                        status = "Awake & Focused"
                        status_color = (0, 255, 0)
        else:
            # Khi mặt lọt ra ngoài, tạm dừng kiểm tra mắt để tập trung xoay camera
            sleep_counter = 0
            yawn_counter = 0
            status = "Adjusting Camera..."
            status_color = (0, 255, 255) # Màu vàng
                
        # 3. HIỂN THỊ THÔNG SỐ LÊN MÀN HÌNH
        # Đẩy EAR/MAR và Status xuống dưới để không đè vào Pan/Tilt
        cv2.putText(frame, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"STATUS: {status}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # FPS đẩy sang góc trên bên phải
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Render
        cv2.imshow("Robot Vision System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Dọn dẹp
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()