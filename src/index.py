import time
import math
import cv2
import mediapipe as mp

# ==========================================
# 1. CÁC HÀM TÍNH TOÁN TOÁN HỌC (EAR & MAR)
# ==========================================
def calculate_distance(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm (x, y)"""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def calculate_ear(eye_indices, landmarks, w, h):
    """Tính Eye Aspect Ratio (EAR) để đo độ mở của mắt"""
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    v1 = calculate_distance(eye_points[1], eye_points[5])
    v2 = calculate_distance(eye_points[2], eye_points[4])
    h_dist = calculate_distance(eye_points[0], eye_points[3])
    
    return (v1 + v2) / (2.0 * h_dist) if h_dist != 0 else 0

def calculate_mar(mouth_indices, landmarks, w, h):
    """Tính Mouth Aspect Ratio (MAR) để đo độ mở của miệng (ngáp)"""
    p_top = (int(landmarks[mouth_indices[0]].x * w), int(landmarks[mouth_indices[0]].y * h))
    p_bottom = (int(landmarks[mouth_indices[1]].x * w), int(landmarks[mouth_indices[1]].y * h))
    p_left = (int(landmarks[mouth_indices[2]].x * w), int(landmarks[mouth_indices[2]].y * h))
    p_right = (int(landmarks[mouth_indices[3]].x * w), int(landmarks[mouth_indices[3]].y * h))
    
    v_dist = calculate_distance(p_top, p_bottom)
    h_dist = calculate_distance(p_left, p_right)
    
    return v_dist / h_dist if h_dist != 0 else 0


# ==========================================
# 2. KHỞI TẠO THÔNG SỐ VÀ MEDIA PIPE
# ==========================================
# Khởi tạo Video/Camera (Thay "data/video.mp4" thành 0 nếu dùng webcam thật)
cap = cv2.VideoCapture("data/video.mp4")
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video", 800, 600)
pTime = 0

# Khởi tạo MediaPipe Face Mesh
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Định nghĩa các điểm mốc (Landmarks) để tính toán
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [13, 14, 78, 308] # Trên, Dưới, Trái, Phải

# Cấu hình Ngưỡng (Thresholds) & Bộ đếm
EAR_THRESHOLD = 0.22  # Dưới mức này là nhắm mắt
MAR_THRESHOLD = 0.5   # Trên mức này là ngáp
FRAMES_TO_SLEEP = 15  # Số frame liên tiếp nhắm mắt để cảnh báo ngủ
FRAMES_TO_YAWN = 10   # Số frame liên tiếp mở miệng to để cảnh báo ngáp

sleep_counter = 0
yawn_counter = 0
status = "Awake & Focused"
status_color = (0, 255, 0) # Xanh lá

# ==========================================
# 3. VÒNG LẶP XỬ LÝ CHÍNH
# ==========================================
while True:
    success, frame = cap.read()
    if not success:
        print("Video ended or cannot be read.")
        break
    
    h, w, c = frame.shape
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(frameRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            
            # --- PHẦN 1: VẼ ĐỒ HỌA MÀU NỔI BẬT ---
            # Bút vẽ màu xanh (cho viền/mạng lưới khuôn mặt)
            spec_green = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            # Bút vẽ màu đỏ (nhấn mạnh mắt và môi)
            spec_red = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)

            # Vẽ toàn bộ viền mặt (Màu xanh)
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_CONTOURS, spec_green, spec_green)
            # Vẽ đè Mắt trái (Màu đỏ)
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LEFT_EYE, spec_red, spec_red)
            # Vẽ đè Mắt phải (Màu đỏ)
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_RIGHT_EYE, spec_red, spec_red)
            # Vẽ đè Môi (Màu đỏ)
            mpDraw.draw_landmarks(frame, faceLms, mpFaceMesh.FACEMESH_LIPS, spec_red, spec_red)


            # --- PHẦN 2: TÍNH TOÁN LOGIC BUỒN NGỦ & NGÁP ---
            landmarks = faceLms.landmark
            
            left_ear = calculate_ear(LEFT_EYE_INDICES, landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE_INDICES, landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(MOUTH_INDICES, landmarks, w, h)
            
            # Logic kiểm tra Ngáp (Yawning)
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= FRAMES_TO_YAWN:
                    status = "Yawning (Fatigue!)"
                    status_color = (0, 165, 255) # Màu cam
            else:
                yawn_counter = 0
            
            # Logic kiểm tra Nhắm mắt (Sleeping) - Ưu tiên cao hơn
            if avg_ear < EAR_THRESHOLD:
                sleep_counter += 1
                if sleep_counter >= FRAMES_TO_SLEEP:
                    status = "Sleeping (Drowsy!)"
                    status_color = (0, 0, 255) # Màu đỏ
            else:
                sleep_counter = 0
                
            # Trở lại trạng thái bình thường
            if sleep_counter == 0 and yawn_counter == 0:
                status = "Awake & Focused"
                status_color = (0, 255, 0)
            
            # --- PHẦN 3: HIỂN THỊ THÔNG SỐ LÊN MÀN HÌNH ---
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"STATUS: {status}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

    # Tính toán và hiển thị FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
    pTime = cTime
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Hiển thị cửa sổ
    cv2.imshow("Video", frame)
    
    # Nhấn 'q' để thoát an toàn
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Dọn dẹp bộ nhớ sau khi tắt
cap.release()
cv2.destroyAllWindows()