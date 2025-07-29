import cv2
import pyautogui
import sys
import os
import mediapipe as mp

# GazeTrackingフォルダのパスを設定
current_dir = os.path.dirname(os.path.abspath(__file__))
gaze_tracking_path = os.path.join(current_dir, 'GazeTracking')
sys.path.append(gaze_tracking_path)
from gaze_tracking import GazeTracking

# --- 設定値 ---
CALIBRATION_FRAMES = 120
#ALPHA = 0.5 # 急激な視線の変化を抑えてなめらかにする
ALPHA = 0.9 # スムージング最小（敏感だが重さが減る）

def remap_value(value, in_min, in_max):
    if in_max == in_min:
        return 0.5
    scaled_value = (value - in_min) / (in_max - in_min)
    return max(0.0, min(1.0, scaled_value))

# --- 初期化 ---
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()
smooth_x, smooth_y = screen_w / 2, screen_h / 2
calibrated = False
frame_count = 0
h_min, h_max = 1.0, 0.0
v_min, v_max = 1.0, 0.0

# MediaPipe Hands 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

print("キャリブレーションを開始します。5秒間、画面の四隅をゆっくりと見てください...")

def is_peace_sign(hand_landmarks):
    # 指の先端の座標 (y値が小さいほど上にある)
    tips = [8, 12]  # 人差し指、中指
    folded = [16, 20]  # 薬指、小指

    def is_extended(tip_id, pip_id):
        return hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y

    return (
        is_extended(8, 6) and  # 人差し指
        is_extended(12, 10) and  # 中指
        not is_extended(16, 14) and  # 薬指は曲がっている
        not is_extended(20, 18)  # 小指も曲がっている
    )

while True:
    success, frame = webcam.read()
    if not success:
        continue
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- MediaPipeで手を検出 ---
    results = hands.process(rgb_frame)
    peace_detected = False
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_peace_sign(hand_landmarks):
                peace_detected = True
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- 視線情報の処理 ---
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    if not calibrated:
        text = f"Calibrating... {CALIBRATION_FRAMES - frame_count}"
        cv2.putText(annotated_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        if hr is not None and vr is not None:
            h_min = min(h_min, hr)
            h_max = max(h_max, hr)
            v_min = min(v_min, vr)
            v_max = max(v_max, vr)

        frame_count += 1
        if frame_count >= CALIBRATION_FRAMES:
            calibrated = True
            print("キャリブレーション完了！")
            print(f"視線範囲 - H: {h_min:.2f}-{h_max:.2f}, V: {v_min:.2f}-{v_max:.2f}")

    else:
        if gaze.pupils_located:
            hr = gaze.horizontal_ratio()
            vr = gaze.vertical_ratio()

            if hr is not None and vr is not None:
                norm_h = remap_value(hr, h_min, h_max)
                norm_v = remap_value(vr, v_min, v_max)
                text = f"{hr:.2f}, {vr:.2f}"
                cv2.putText(annotated_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

                target_x = screen_w * norm_h
                target_y = screen_h * norm_v

                smooth_x = ALPHA * target_x + (1 - ALPHA) * smooth_x
                smooth_y = ALPHA * target_y + (1 - ALPHA) * smooth_y

                pyautogui.moveTo(smooth_x, smooth_y)

        # Peace sign を検出したら表示
        if peace_detected:
            cv2.putText(annotated_frame, "Peace Sign Detected", (60, 120),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow("Gaze + Peace Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
