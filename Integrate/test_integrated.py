import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# --- 初期化 ---

# 画面サイズの取得
screen_w, screen_h = pyautogui.size()

# MediaPipeのモデルを初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils

# Webカメラの初期化
cap = cv2.VideoCapture(0)

# --- 視線追跡用の変数 ---
calibration_points = []
gaze_min = [1.0, 1.0]
gaze_max = [0.0, 0.0]
alpha = 0.3
calibrated = False

# キャリブレーション用のガイドテキスト
guide_texts = [
    "Look at the TOP LEFT corner and press SPACE",
    "Look at the TOP RIGHT corner and press SPACE",
    "Look at the BOTTOM LEFT corner and press SPACE",
    "Look at the BOTTOM RIGHT corner and press SPACE",
    "Look at the CENTER and press SPACE"
]

# --- ハンドジェスチャー用の変数 ---
gesture_state = "Ready"
current_action = "None"
scroll_cooldown = 0
dragging = False
tips_ids = [4, 8, 12, 16, 20] # 各指先のランドマークID

# --- ヘルパー関数: 指の開閉を検出 ---
def detect_fingers_up(landmarks):
    """
    手のランドマーク情報から、各指が上がっているか（開いているか）を判定する関数
    """
    fingers = []
    # 親指: x座標で判定（手の向きに依存するため単純化）
    # 親指の先端(4)が、その付け根に近い点(3)より外側にあれば開いていると判定
    if landmarks.landmark[tips_ids[0]].x < landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(True)
    else:
        fingers.append(False)

    # 人差し指から小指: y座標で判定
    # 各指の先端が、2つ下の関節よりも上にあれば開いていると判定
    for i in range(1, 5):
        if landmarks.landmark[tips_ids[i]].y < landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(True)
        else:
            fingers.append(False)
    return fingers

# --- メイン処理 ---

print("🔧 キャリブレーションを開始します。")
print("Webカメラのウィンドウをアクティブにして、画面の指示に従ってください。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 映像を左右反転し、処理のためにBGRからRGBに変換
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # 顔と手のランドマークを検出
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # --- 1. 視線追跡とカーソル移動 ---
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # 右目の瞳孔のランドマーク(473)を使用
        gaze_x = face_landmarks.landmark[473].x
        gaze_y = face_landmarks.landmark[473].y

        if not calibrated:
            # --- キャリブレーションフェーズ ---
            idx = len(calibration_points)
            if idx < 5:
                # 画面にガイドテキストを表示
                guide_text = guide_texts[idx]
                cv2.putText(frame, guide_text, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # スペースキーが押されたら
                calibration_points.append((gaze_x, gaze_y))
                print(f"📌 キャリブレーションポイント {idx+1}/5 を記録しました。")
                
                if len(calibration_points) == 5:
                    # 5点集まったらキャリブレーション完了
                    gaze_min[0] = min(p[0] for p in calibration_points)
                    gaze_max[0] = max(p[0] for p in calibration_points)
                    gaze_min[1] = min(p[1] for p in calibration_points)
                    gaze_max[1] = max(p[1] for p in calibration_points)
                    calibrated = True
                    print("✅ キャリブレーション完了！ マウス操作を開始します。")
                    # キャリブレーションガイドを消すために少し待つ
                    cv2.waitKey(500) 
        
        else:
            # --- マウス移動フェーズ ---
            # 視線座標を画面座標にマッピング
            # 範囲外の値が出ないようにクリップ
            norm_x = np.clip((gaze_x - gaze_min[0]) / (gaze_max[0] - gaze_min[0]), 0, 1)
            norm_y = np.clip((gaze_y - gaze_min[1]) / (gaze_max[1] - gaze_min[1]), 0, 1)
            
            screen_x = screen_w * norm_x
            screen_y = screen_h * norm_y

            # カーソルの動きを滑らかにする (スムージング)
            prev_x, prev_y = pyautogui.position()
            # alpha = 0.3 # スムージング係数（小さいほど滑らか）
            smooth_x = prev_x * (1 - alpha) + screen_x * alpha
            smooth_y = prev_y * (1 - alpha) + screen_y * alpha

            # ドラッグ中でなければカーソルを移動
            pyautogui.moveTo(smooth_x, smooth_y)
            
            # デバッグ用に視線位置を円で表示
            cv2.circle(frame, (int(gaze_x * img_w), int(gaze_y * img_h)), 5, (0, 255, 0), -1)

    # --- 2. ハンドジェスチャーによる操作 (キャリブレーション完了後のみ) ---
    current_action = "None"
    if calibrated and hand_results.multi_hand_landmarks:
        for handLms in hand_results.multi_hand_landmarks:
            # ジェスチャー認識の前に現在の状態をリセット
            gesture_state = "Ready"

            fingers = detect_fingers_up(handLms)

            # --- ジェスチャー判定 ---
            if fingers == [True, False, False, False, False]:  # 親指のみ -> 上スクロール
                current_action = "Scroll Up"
                if scroll_cooldown == 0:
                    pyautogui.scroll(10)
                    scroll_cooldown = 10 # クールダウン設定

            elif fingers == [False, False, False, False, True]: # 小指のみ -> 下スクロール
                current_action = "Scroll Down"
                if scroll_cooldown == 0:
                    pyautogui.scroll(-10)
                    scroll_cooldown = 10 # クールダウン設定

            elif fingers == [False, True, False, False, False]: # 人差し指のみ -> クリック
                current_action = "Click"
                if gesture_state != "click":
                    pyautogui.click()
                    gesture_state = "click"

            elif fingers == [False, True, True, False, False]:  # 人差し指と中指 -> ダブルクリック
                current_action = "Double Click"
                if gesture_state != "double":
                    pyautogui.doubleClick()
                    gesture_state = "double"
            
            elif fingers == [True, True, False, False, False]: # 親指と人差し指 -> 右クリック
                current_action = "Right Click"
                if gesture_state != "right":
                    pyautogui.rightClick()
                    gesture_state = "right"

            elif fingers == [False, False, False, False, False]:  # グー -> ドラッグ開始
                current_action = "Dragging"
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                gesture_state = "drag"

            elif fingers == [True, True, True, True, True]:  # パー -> 基本状態 / ドラッグ終了
                current_action = "Ready"
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False
                gesture_state = "Ready"

            # 手のランドマークを描画
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
    
    if current_action == "None": alpha = 0.3
    else: alpha = 0.1
    
    # スクロールのクールダウン処理
    if scroll_cooldown > 0:
        scroll_cooldown -= 1

    # --- 画面表示 ---
    if calibrated:
        cv2.putText(frame, f"Action: {current_action}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Gaze and Gesture Mouse Control", frame)
    
    # 'q'キーまたはESCキーで終了
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# --- 終了処理 ---
cap.release()
cv2.destroyAllWindows()