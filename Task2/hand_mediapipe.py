import cv2
import mediapipe as mp
import pyautogui
import time

# 初期化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

gesture_state = "Ready"
scroll_cooldown = 0
dragging = False
tips_ids = [4, 8, 12, 16, 20]

def detect_fingers_up(landmarks):
    fingers = []
    fingers.append(landmarks.landmark[4].x < landmarks.landmark[3].x)  # 親指（横向き）
    for i in range(1, 5):  # 他の指（縦向き）
        fingers.append(landmarks.landmark[tips_ids[i]].y < landmarks.landmark[tips_ids[i] - 2].y)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_action = "None"

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            fingers = detect_fingers_up(handLms)

            # ジェスチャー判定
            if fingers == [True, False, False, False, False]:  # 上スクロール
                current_action = "Scroll Up"
                if scroll_cooldown == 0:
                    pyautogui.scroll(100)
                    scroll_cooldown = 10

            elif fingers == [False, False, False, False, True]:  #下スクロール
                current_action = "Scroll Down"
                if scroll_cooldown == 0:
                    pyautogui.scroll(-100)
                    scroll_cooldown = 10

            elif fingers == [False, True, False, False, False]:  # 一本指 → クリック
                current_action = "Click"
                if gesture_state != "click":
                    pyautogui.click()
                    gesture_state = "click"

            elif fingers == [False, True, True, False, False]:  # チョキ → ダブルクリック
                current_action = "Double Click"
                if gesture_state != "double":
                    pyautogui.doubleClick()
                    gesture_state = "double"

            elif fingers == [True, True, False, False, False]: # 親指＋人差し指　→　右クリック
                current_action = "Right Click"
                if gesture_state != "right":
                    pyautogui.rightClick()
                    gesture_state = "right"


            elif fingers == [False, False, False, False, False]:  # グー → ドラッグ
                current_action = "Dragging"
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                gesture_state = "drag"

            elif fingers == [True, True, True, True, True]:  # パー → 基本状態
                current_action = "Ready"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                gesture_state = "Ready"

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    if scroll_cooldown > 0:
        scroll_cooldown -= 1

    # 表示
    cv2.putText(frame, f"Action: {current_action}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
