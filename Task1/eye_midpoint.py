import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# 画面サイズ取得
screen_w, screen_h = pyautogui.size()

# Mediapipe FaceMesh 初期化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# カメラ入力
cap = cv2.VideoCapture(0)

# キャリブレーション変数
calibration_points = []
gaze_min = [1, 1]
gaze_max = [0, 0]
calibrated = False

# キャリブレーション用のガイド表示テキスト
guide_texts = [
    "Look at the TOP LEFT corner and press SPACE",
    "Look at the TOP RIGHT corner and press SPACE",
    "Look at the BOTTOM LEFT corner and press SPACE",
    "Look at the BOTTOM RIGHT corner and press SPACE",
    "Look at the CENTER and press SPACE"
]

print("🔧 キャリブレーションを開始します。")
print("画面の左上・右上・左下・右下・中央を見て、スペースキーを押してください。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # 👇 ランドマークを「眉間(168)」に変更
        gaze_x = face_landmarks.landmark[168].x
        gaze_y = face_landmarks.landmark[168].y

        if not calibrated:
            idx = len(calibration_points)

            # キャリブレーションガイド表示
            if idx < 5:
                guide_text = guide_texts[idx]
                cv2.putText(frame, guide_text, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            key = cv2.waitKey(1)
            if key == 32:  # スペースキー
                calibration_points.append((gaze_x, gaze_y))
                print(f"📌 キャリブレーションポイント記録: {gaze_x:.3f}, {gaze_y:.3f}")
                if len(calibration_points) == 5:
                    gaze_min[0] = min(p[0] for p in calibration_points)
                    gaze_max[0] = max(p[0] for p in calibration_points)
                    gaze_min[1] = min(p[1] for p in calibration_points)
                    gaze_max[1] = max(p[1] for p in calibration_points)
                    calibrated = True
                    print("✅ キャリブレーション完了！")
        else:
            # キャリブレーション済みなら、補正してマウス移動
            norm_x = (gaze_x - gaze_min[0]) / (gaze_max[0] - gaze_min[0])
            norm_y = (gaze_y - gaze_min[1]) / (gaze_max[1] - gaze_min[1])
            screen_x = screen_w * np.clip(norm_x, 0, 1)
            screen_y = screen_h * np.clip(norm_y, 0, 1)

            # スムージング処理
            prev_x, prev_y = pyautogui.position()
            alpha = 0.15
            smooth_x = prev_x * (1 - alpha) + screen_x * alpha
            smooth_y = prev_y * (1 - alpha) + screen_y * alpha

            pyautogui.moveTo(smooth_x, smooth_y)

            # デバッグ描画（眉間位置を緑丸で表示）
            cv2.circle(frame, (int(gaze_x * img_w), int(gaze_y * img_h)), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"Gaze: ({int(screen_x)}, {int(screen_y)})",
                        (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Gaze Tracker (Midpoint)", frame)
    if cv2.waitKey(1) == 27:  # ESCで終了
        break

cap.release()
cv2.destroyAllWindows()
