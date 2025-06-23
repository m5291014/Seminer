import cv2
import mediapipe as mp

# MediaPipe Handsのセットアップ
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# 指の先端と第2関節のランドマークID
FINGER_TIPS = [8, 12, 16, 20]  # 人差し指〜小指の先端
FINGER_PIPS = [6, 10, 14, 18]  # 第2関節

# ✌️ピースサイン判定
def is_peace(landmarks):
    count = 0
    for tip_id, pip_id in zip(FINGER_TIPS, FINGER_PIPS):
        if landmarks[tip_id].y < landmarks[pip_id].y:
            count += 1
    return count == 2

# 利用可能なUSBカメラを探す（iPhone連携を除外）
def find_working_usb_camera(max_index=5):
    print("USBカメラをスキャン中...")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 解像度をチェックしてiPhoneを除外（iPhoneは独特のサイズを持つことが多い）
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            if width > 0 and height > 0 and width <= 1920:
                print(f"USBカメラ {i} を使用します（解像度: {int(width)}x{int(height)}）")
                return cap
            cap.release()
    print("利用可能なUSBカメラが見つかりませんでした。")
    return None

# カメラ初期化
cap = find_working_usb_camera()

if cap is None:
    exit()

print("✌️ ピースサインを認識中（ESCキーで終了）")

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得に失敗しました。")
        break

    frame = cv2.flip(frame, 1)  # 鏡表示
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_peace(hand_landmarks.landmark):
                cv2.putText(frame, 'Peace Sign ✌️', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Peace Sign', (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow('USB Camera - Hand Sign Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
        break

# 終了処理
cap.release()
cv2.destroyAllWindows()
