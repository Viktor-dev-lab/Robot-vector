import math

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