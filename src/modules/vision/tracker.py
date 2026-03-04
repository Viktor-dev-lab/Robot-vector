import cv2

class PanTiltTracker:
    # edge_margin: khoảng cách từ viền camera vào trong để tạo vùng an toàn (Safe Zone)
    def __init__(self, input_w=640, input_h=480, edge_margin=80):
        self.input_w = input_w
        self.input_h = input_h
        self.edge_margin = edge_margin

    def process_and_draw(self, frame, faces):
        # Mặc định luôn là CENTER, giải quyết luôn yêu cầu "không có mặt thì mặc định là center"
        pan_cmd, tilt_cmd = "Pan: CENTER", "Tilt: CENTER"
        
        # 1. Vẽ khung Safe Zone (Khu vực an toàn)
        # Khuôn mặt nằm trọn trong ô xanh này thì robot đứng im, không quay
        cv2.rectangle(frame, (self.edge_margin, self.edge_margin), 
                             (self.input_w - self.edge_margin, self.input_h - self.edge_margin), 
                             (255, 0, 0), 2)

        # 2. Xử lý logic khi nhận diện được khuôn mặt
        # Kiểm tra faces không bị None và không rỗng
        if faces is not None and len(faces) > 0:
            # Lấy mặt to nhất/đầu tiên làm mục tiêu
            main_face = faces[0] 
            box = list(map(int, main_face[:4]))
            fx, fy, fw, fh = box
        
            
            # --- LOGIC MỚI: Tính khoảng cách từ viền mặt đến viền Frame ---
            # Nếu viền trái của mặt chạm/vượt viền trái của Safe Zone -> Quay trái
            if fx < self.edge_margin: 
                pan_cmd = "Pan: LEFT (<-)"
            # Nếu viền phải của mặt chạm/vượt viền phải của Safe Zone -> Quay phải
            elif fx + fw > self.input_w - self.edge_margin: 
                pan_cmd = "Pan: RIGHT (->)"
                
            # Nếu viền trên của mặt chạm/vượt viền trên của Safe Zone -> Ngẩng lên
            if fy < self.edge_margin: 
                tilt_cmd = "Tilt: UP (^)"
            # Nếu viền dưới của mặt chạm/vượt viền dưới của Safe Zone -> Cúi xuống
            elif fy + fh > self.input_h - self.edge_margin: 
                tilt_cmd = "Tilt: DOWN (v)"

            # Hiển thị thông số khoảng cách tới viền (dùng để bạn canh chỉnh margin)
            dist_left = fx
            dist_right = self.input_w - (fx + fw)
            cv2.putText(frame, f"Dist L: {dist_left} | Dist R: {dist_right}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Vẽ landmarks nếu model của bạn trả về > 4 giá trị (vd: YOLO-Face)
            if len(main_face) >= 14:
                landmarks = list(map(int, main_face[4:14]))
                for i in range(5):
                    cv2.circle(frame, (landmarks[i*2], landmarks[i*2+1]), 3, (0, 0, 255), -1)

        # 3. In lệnh Command ra màn hình (hoạt động kể cả khi có hoặc không có mặt)
        cv2.putText(frame, pan_cmd, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, tilt_cmd, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, pan_cmd, tilt_cmd