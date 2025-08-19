import cv2
import mediapipe as mp
import pyautogui

# ===== パラメータ（調整用） =====
VOLUME_CD = 10          # 連射抑制フレーム数
NOSE_TOL_Y = 0.12       # 鼻y±許容帯（顔の高さに対する割合。広げる→値を大きく）

# ===== MediaPipe 初期化 =====
mp_hands = mp.solutions.hands
mp_fd = mp.solutions.face_detection
hands = mp_hands.Hands(max_num_hands=1)
face = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ランドマークID
TIPS = [4, 8, 12, 16, 20]          # 親指〜小指の先
MCPs = [5, 9, 13, 17]           # 親指MCP, 人差し指MCP, 中指MCP, 薬指MCP, 小指MCP

volume_cooldown = 0

def fingers_up(lm):
    """[thumb, index, middle, ring, pinky] True/False を返す"""
    f = []
    # 親指（右手想定。左手で誤る場合は < を > に反転）
    f.append(lm.landmark[4].x < lm.landmark[3].x)
    # 他の指：tip が PIP より上（y が小さい）なら True
    for tid in TIPS[1:]:
        f.append(lm.landmark[tid].y < lm.landmark[tid-2].y)
    return f

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_res = hands.process(rgb)
    face_res = face.process(rgb)

    current_action = "None"

    # --- 顔（1人前提）から鼻の y と顔の高さを取得 ---
    nose_y = None
    face_h = None
    if face_res.detections:
        det = face_res.detections[0]
        rbb = det.location_data.relative_bounding_box
        y0 = int(rbb.ymin * h); y1 = int((rbb.ymin + rbb.height) * h)
        face_h = max(1, y1 - y0)
        if det.location_data.relative_keypoints:
            # keypoints: 0右目,1左目,2鼻,3口,4右耳珠,5左耳珠（モデルによる）
            nose = det.location_data.relative_keypoints[2]
            nose_y = nose.y * h

    if hand_res.multi_hand_landmarks and (nose_y is not None) and (face_h is not None):
        tol = face_h * NOSE_TOL_Y
        top = nose_y - tol
        bot = nose_y + tol

        for handLms in hand_res.multi_hand_landmarks:
            fvec = fingers_up(handLms)

            # === 条件：指の付け根（MCP）のどれか1本でも鼻の高さ帯に入っている（xは見ない）===
            mcp_y_hits = []
            for mcp_id in MCPs:
                mcp_y = handLms.landmark[mcp_id].y * h
                mcp_y_hits.append(top <= mcp_y <= bot)

            near_nose_height = any(mcp_y_hits)

            if near_nose_height:
                # パー（全指True）→ 音量UP
                if fvec == [True, True, True, True, True]:
                    current_action = "Volume Up"
                    if volume_cooldown == 0:
                        pyautogui.press('volumeup')
                        volume_cooldown = VOLUME_CD

                # 人差し指のみ → 音量DOWN
                elif fvec == [False, True, False, False, False]:
                    current_action = "Volume Down"
                    if volume_cooldown == 0:
                        pyautogui.press('volumedown')
                        volume_cooldown = VOLUME_CD

            # 可視化（手の骨格）
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # クールダウン
    if volume_cooldown > 0:
        volume_cooldown -= 1

    # デバッグ描画（鼻高さ帯の2本線）
    if nose_y is not None and face_h is not None:
        tol = int(face_h * NOSE_TOL_Y)
        y_top = int(nose_y - tol); y_bot = int(nose_y + tol)
        cv2.line(frame, (0, y_top), (w, y_top), (120, 180, 120), 1)
        cv2.line(frame, (0, y_bot), (w, y_bot), (120, 180, 120), 1)

    cv2.putText(frame, f"Action: {current_action}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Volume by Nose-Height MCP (Y-only)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
