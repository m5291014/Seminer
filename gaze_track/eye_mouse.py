import cv2
import pyautogui
import sys
import os

# GazeTrackingフォルダへの絶対パスを追加
# このファイル(test.py)があるディレクトリを取得し、GazeTrackingを結合
current_dir = os.path.dirname(os.path.abspath(__file__))
gaze_tracking_path = os.path.join(current_dir, 'GazeTracking')
sys.path.append(gaze_tracking_path)

from gaze_tracking import GazeTracking


# --- 設定値 ---
# キャリブレーション時間（フレーム数）。60フレーム=約2秒
CALIBRATION_FRAMES = 120
# スムージングの度合い（0に近いほど滑らか）
ALPHA = 0.5

def remap_value(value, in_min, in_max):
    """入力値が入力範囲のどの割合かを計算し、0.0-1.0の範囲で返す"""
    # ゼロ除算を防止
    if in_max == in_min:
        return 0.5

    # 割合を計算し、0.0-1.0の範囲に収める
    scaled_value = (value - in_min) / (in_max - in_min)
    return max(0.0, min(1.0, scaled_value))

# --- 初期化 ---
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

# スムージング用の変数
smooth_x, smooth_y = screen_w / 2, screen_h / 2

# キャリブレーション用の変数
calibrated = False
frame_count = 0
h_min, h_max = 1.0, 0.0
v_min, v_max = 1.0, 0.0


print("キャリブレーションを開始します。5秒間、画面の四隅をゆっくりと見てください...")

while True:
    _, frame = webcam.read()
    if frame is None:
        continue
    frame = cv2.flip(frame, 1)
    gaze.refresh(frame)
    annotated_frame = gaze.annotated_frame()

    # --- キャリブレーションフェーズ ---
    if not calibrated:
        text = f"Calibrating... {CALIBRATION_FRAMES - frame_count}"
        cv2.putText(annotated_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        
        hr = gaze.horizontal_ratio()
        vr = gaze.vertical_ratio()

        if hr is not None and vr is not None:
            # 水平・垂直方向の視線範囲（最小・最大）を記録
            h_min = min(h_min, hr)
            h_max = max(h_max, hr)
            v_min = min(v_min, vr)
            v_max = max(v_max, vr)

        frame_count += 1
        if frame_count >= CALIBRATION_FRAMES:
            calibrated = True
            print("キャリブレーション完了！")
            print(f"認識された視線範囲 - H: {h_min:.2f}-{h_max:.2f}, V: {v_min:.2f}-{v_max:.2f}")

    # --- マウスコントロールフェーズ ---
    else:
        if gaze.pupils_located:
            hr = gaze.horizontal_ratio()
            vr = gaze.vertical_ratio()

            if hr is not None and vr is not None:
                # キャリブレーション結果を元に、現在の視線位置を0.0-1.0に正規化
                norm_h = remap_value(hr, h_min, h_max)
                norm_v = remap_value(vr, v_min, v_max)
                text = f"{hr, vr}"
                cv2.putText(annotated_frame, text, (60, 60), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)

                target_x = screen_w * (norm_h) # 左右反転解除
                target_y = screen_h * norm_v
                
                # スムージング
                smooth_x = ALPHA * target_x + (1 - ALPHA) * smooth_x
                smooth_y = ALPHA * target_y + (1 - ALPHA) * smooth_y

                pyautogui.moveTo(smooth_x, smooth_y)

    cv2.imshow("Gaze Tracking Mouse Control", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()