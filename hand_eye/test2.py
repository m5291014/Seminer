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

# GazeTracking のパス追加（カレントディレクトリ内に GazeTracking フォルダがある場合）
current_dir = os.path.dirname(os.path.abspath(__file__))
gaze_tracking_path = os.path.join(current_dir, 'GazeTracking')
sys.path.append(gaze_tracking_path)

# 初期化
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
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# ハンドジェスチャー関連
gesture_state = "Ready"
scroll_cooldown = 0
dragging = False
tips_ids = [4, 8, 12, 16, 20]

def remap_value(value, in_min, in_max):
    if in_max == in_min:
        return 0.5
    scaled_value = (value - in_min) / (in_max - in_min)
    return max(0.0, min(1.0, scaled_value))

def detect_fingers_up(landmarks):
    fingers = []
    fingers.append(landmarks.landmark[4].x < landmarks.landmark[3].x)  # 親指（横）
    for i in range(1, 5):  # 他の指（縦）
        fingers.append(landmarks.landmark[tips_ids[i]].y < landmarks.landmark[tips_ids[i] - 2].y)
    return fingers

print("キャリブレーション中です。画面の四隅を見ながら5秒待ってください...")

while True:
    success, frame = webcam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 手の検出 ---
    results = hands.process(rgb_frame)
    hand_detected = False
    current_action = "None"

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            hand_detected = True
            fingers = detect_fingers_up(handLms)

            # ジェスチャー処理
            if fingers == [True, False, False, False, False]:
                current_action = "Scroll Up"
                if scroll_cooldown == 0:
                    pyautogui.scroll(100)
                    scroll_cooldown = 10

            elif fingers == [False, False, False, False, True]:
                current_action = "Scroll Down"
                if scroll_cooldown == 0:
                    pyautogui.scroll(-100)
                    scroll_cooldown = 10

            elif fingers == [False, True, False, False, False]:
                current_action = "Click"
                if gesture_state != "click":
                    pyautogui.click()
                    gesture_state = "click"

            elif fingers == [False, True, True, False, False]:
                current_action = "Double Click"
                if gesture_state != "double":
                    pyautogui.doubleClick()
                    gesture_state = "double"

            elif fingers == [True, True, False, False, False]:
                current_action = "Right Click"
                if gesture_state != "right":
                    pyautogui.rightClick()
                    gesture_state = "right"

            elif fingers == [False, False, False, False, False]:
                current_action = "Dragging"
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                gesture_state = "drag"

            elif fingers == [True, True, True, True, True]:
                current_action = "Ready"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                gesture_state = "Ready"

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # --- 視線検出 ---
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    if not calibrated:
        text = f"Calibrating... {CALIBRATION_FRAMES - frame_count}"
        cv2.putText(annotated_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 2)

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

                target_x = screen_w * norm_h
                target_y = screen_h * norm_v

                smooth_x = ALPHA * target_x + (1 - ALPHA) * smooth_x
                smooth_y = ALPHA * target_y + (1 - ALPHA) * smooth_y

                pyautogui.moveTo(smooth_x, smooth_y)

    # クールダウン処理
    if scroll_cooldown > 0:
        scroll_cooldown -= 1

    # 表示
    if hand_detected:
        cv2.putText(annotated_frame, f"Action: {current_action}", (60, 120), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Gaze + Gesture Control", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
