import config
from shares.utils.face_math import calculate_ear, calculate_mar

def analyze_fatigue(frame, results, w, h, sleep_counter, yawn_counter, mp_draw, mp_face_mesh, spec_green, spec_blue):
    status = "Awake & Focused"
    status_color = config.COLOR_GREEN
    status_code = 0
    avg_ear, mar = 0.0, 0.0

    for face_lms in results.multi_face_landmarks:
        mp_draw.draw_landmarks(frame, face_lms, mp_face_mesh.FACEMESH_CONTOURS, spec_green, spec_green)
        mp_draw.draw_landmarks(frame, face_lms, mp_face_mesh.FACEMESH_LEFT_EYE, spec_blue, spec_blue)
        mp_draw.draw_landmarks(frame, face_lms, mp_face_mesh.FACEMESH_RIGHT_EYE, spec_blue, spec_blue)
        mp_draw.draw_landmarks(frame, face_lms, mp_face_mesh.FACEMESH_LIPS, spec_blue, spec_blue)

        landmarks = face_lms.landmark
        left_ear = calculate_ear(config.LEFT_EYE_INDICES, landmarks, w, h)
        right_ear = calculate_ear(config.RIGHT_EYE_INDICES, landmarks, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = calculate_mar(config.MOUTH_INDICES, landmarks, w, h)
        
        if mar > config.MAR_THRESHOLD:
            yawn_counter += 1
            if yawn_counter >= config.FRAMES_TO_YAWN:
                status, status_color, status_code = "Yawning (Fatigue!)", config.COLOR_ORANGE, 2
        else:
            yawn_counter = 0
        
        if avg_ear < config.EAR_THRESHOLD:
            sleep_counter += 1
            if sleep_counter >= config.FRAMES_TO_SLEEP:
                status, status_color, status_code = "Sleeping (Drowsy!)", config.COLOR_RED, 1
        else:
            sleep_counter = 0
            if yawn_counter == 0:
                status, status_color, status_code = "Awake & Focused", config.COLOR_GREEN, 0

    return frame, avg_ear, mar, sleep_counter, yawn_counter, status, status_color, status_code