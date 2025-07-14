import cv2
import pyautogui
import time
import sys
sys.path.append('./GazeTracking')  
from gaze_tracking import GazeTracking
from gaze_tracking import GazeTracking
import mediapipe as mp

# 初期化
gaze = GazeTracking()
cap = cv2.VideoCapture(0)
hands_module = mp.solutions.hands
hands = hands_module.Hands()
drawer = mp.solutions.drawing_utils

prev_index_x = None
is_fullscreen = False
last_action_time = 0

def detect_fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # 親指：横方向に検出
    fingers.append(hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x)

    # 他の指：縦方向
    for tip_id in tips_ids[1:]:
        pip_id = tip_id - 2
        fingers.append(hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y)

    return fingers  # [thumb, index, middle, ring, pinky]

def get_gesture(fingers):
    count = fingers.count(True)
    if count == 0:
        return "rock"
    elif count == 1 and fingers[1]:
        return "one"
    elif count == 2 and fingers[1] and fingers[2]:
        return "scissors"
    elif count == 5:
        return "paper"
    else:
        return "unknown"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)
    gaze_direction = "center"
    if gaze.is_left(): gaze_direction = "left"
    elif gaze.is_right(): gaze_direction = "right"

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    gesture = "なし"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers = detect_fingers_up(hand_landmarks)
            gesture = get_gesture(fingers)

            drawer.draw_landmarks(frame, hand_landmarks, hands_module.HAND_CONNECTIONS)

            # === 各ジェスチャーに対する処理 ===
            current_time = time.time()
            if current_time - last_action_time > 1.0:  # 誤動作防止
                if gesture == "グー":
                    if is_fullscreen:
                        pyautogui.press('space')  # 再生/停止
                    else:
                        pyautogui.click()
                    last_action_time = current_time

                elif gesture == "チョキ":
                    if is_fullscreen:
                        pyautogui.press('esc')
                        is_fullscreen = False
                    else:
                        pyautogui.press('f')
                        is_fullscreen = True
                    last_action_time = current_time

            # === 一本指によるシーク操作 ===
            if gesture == "一本指":
                index_x = hand_landmarks.landmark[8].x
                if prev_index_x is not None:
                    dx = index_x - prev_index_x
                    if abs(dx) > 0.01:
                        n = int(abs(dx) * 50)
                        for _ in range(n):
                            pyautogui.press('right' if dx > 0 else 'left')
                            time.sleep(0.01)
                        last_action_time = current_time
                prev_index_x = index_x
            else:
                prev_index_x = None  # リセット

    # === 表示用の文字情報 ===
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"Gaze: {gaze_direction}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    #cv2.putText(frame, f"Fullscreen: {is_fullscreen}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("YouTube Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
