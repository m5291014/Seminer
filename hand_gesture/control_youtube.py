import cv2
import mediapipe as mp
import pyautogui
import time
import subprocess # ◀◀◀ 追加 (pygetwindowの代わり)
import sys

# --- macOS専用の関数 ---
def get_chrome_active_url():
    """
    AppleScriptを使ってGoogle ChromeのアクティブなタブのURLを取得する。
    macOS専用。
    """
    # 現在のOSがmacOS ('darwin')でなければ、関数を終了
    if sys.platform != "darwin":
        return ""
        
    script = '''
    if application "Google Chrome" is running then
        tell application "Google Chrome"
            return URL of active tab of front window
        end tell
    end if
    return ""
    '''
    try:
        # AppleScriptを実行し、結果（URL）を取得
        result = subprocess.check_output(['osascript', '-e', script])
        return result.decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Chromeが起動していない、またはその他のエラー
        return ""

# --- 設定 ---
GESTURE_COOLDOWN = 2.0 

# --- 初期化 ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("エラー: カメラを開けませんでした。")
    exit()

last_gesture_time = 0

print("ジェスチャー検出を開始します。YouTubeのウィンドウをアクティブにしてください。")
print("終了するには 'q' キーを押してください。")

# --- メインループ ---
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        
        current_gesture = "none"

        if results.multi_hand_landmarks:
            # （ジェスチャー判定ロジックは変更なしのため省略）
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
                thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
                index_finger_up = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y
                middle_finger_up = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
                ring_finger_up = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y
                pinky_up = landmarks[mp_hands.HandLandmark.PINKY_TIP].y < landmarks[mp_hands.HandLandmark.PINKY_PIP].y
                all_fingers_up = index_finger_up and middle_finger_up and ring_finger_up and pinky_up
                all_fingers_down = not index_finger_up and not middle_finger_up and not ring_finger_up and not pinky_up
                
                if all_fingers_up: current_gesture = "stop"
                elif thumb_tip.y < thumb_ip.y and all_fingers_down: current_gesture = "like"
                elif thumb_tip.y > thumb_ip.y and all_fingers_down: current_gesture = "dislike"
                elif index_finger_up and pinky_up and not middle_finger_up and not ring_finger_up: current_gesture = "rock"

        # 検出結果を映像に描画（省略）
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        
        current_time = time.time()
        if current_gesture != "none" and (current_time - last_gesture_time) > GESTURE_COOLDOWN:
            
            current_url = get_chrome_active_url()
            
            # URLに "youtube.com" が含まれているかチェック
            if "youtube.com" in current_url:
                if current_gesture == "like" or current_gesture == "stop":
                    pyautogui.press('space')
                    print(f"ジェスチャー: {current_gesture} @YouTube -> アクション: 再生/停止 (space)")
                elif current_gesture == "rock":
                    pyautogui.press('l')
                    print(f"ジェスチャー: {current_gesture} @YouTube -> アクション: 10秒早送り (l)")
                elif current_gesture == "dislike":
                    pyautogui.press('j')
                    print(f"ジェスチャー: {current_gesture} @YouTube -> アクション: 10秒戻る (j)")
                
                last_gesture_time = current_time
            else:
                print(f"YouTubeタブがアクティブではありません。 (現在のURL: {current_url})")
            # ◀◀◀ ここまで修正部分 ◀◀◀

        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Hand Gesture YouTube Controller', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
finally:
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("プログラムを終了しました。")