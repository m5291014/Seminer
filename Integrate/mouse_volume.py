import cv2
import mediapipe as mp
import pyautogui

# ===== パラメータ（調整用） =====
# Volume系
VOLUME_CD = 10          # 連射抑制フレーム数
NOSE_TOL_Y = 0.12       # 鼻/耳 y±許容帯（顔の高さに対する割合）
USE_EAR_HEIGHT_IF_AVAILABLE = True  # 耳キーがあれば鼻ではなく耳の高さを使う

# マウス操作系
SCROLL_CD = 10          # スクロールの連射抑制
SCROLL_STEP = 100       # スクロール量

# ===== MediaPipe 初期化 =====
mp_hands = mp.solutions.hands
mp_fd = mp.solutions.face_detection
hands = mp_hands.Hands(max_num_hands=1)
face = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ランドマークID
TIPS = [4, 8, 12, 16, 20]          # 親指〜小指の先
MCPs = [5, 9, 13, 17]              # 人差し/中指/薬指/小指のMCP（親指は除外）

# ステート
gesture_state = "Ready"
dragging = False
scroll_cooldown = 0
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

    # --- 顔情報の取得 ---
    ref_y = None
    face_h = None
    nose_x = None
    y0 = y1 = None

    if face_res.detections:
        det = face_res.detections[0]
        rbb = det.location_data.relative_bounding_box
        y0 = int(rbb.ymin * h); y1 = int((rbb.ymin + rbb.height) * h)
        face_h = max(1, y1 - y0)

        kps = det.location_data.relative_keypoints
        if kps:
            # 鼻の座標（xはゲート用、yはフォールバック高さ）
            nose = kps[2]
            nose_x = nose.x * w

            # 耳の高さを優先、無ければ鼻の高さ
            use_ear = False
            if USE_EAR_HEIGHT_IF_AVAILABLE and len(kps) >= 6:
                right_ear_y = kps[4].y * h if kps[4] is not None else None
                left_ear_y  = kps[5].y * h if kps[5] is not None else None
                if (right_ear_y is not None) and (left_ear_y is not None):
                    ref_y = (right_ear_y + left_ear_y) / 2.0
                    use_ear = True
            if not use_ear:
                ref_y = nose.y * h

    # ===== 手の処理 =====
    if hand_res.multi_hand_landmarks:
        for handLms in hand_res.multi_hand_landmarks:
            fvec = fingers_up(handLms)
            fvec_wo_thumb = fvec[1:]

            # 高さ帯は「音量」だけで使用
            tol = face_h * NOSE_TOL_Y if (ref_y is not None and face_h is not None) else None
            if tol is not None:
                top = ref_y - tol
                bot = ref_y + tol

            # --- (A) 音量（鼻より左 ＆ 高さ帯あり） ---
            did_volume = False
            if (tol is not None) and (nose_x is not None):
                mcp_hits_left = []
                for mcp_id in MCPs:
                    mcp_y = handLms.landmark[mcp_id].y * h
                    mcp_x = handLms.landmark[mcp_id].x * w
                    in_height = (top <= mcp_y <= bot)
                    left_of_nose = (mcp_x < nose_x)
                    mcp_hits_left.append(in_height and left_of_nose)

                if any(mcp_hits_left):
                    if fvec_wo_thumb == [True, True, True, True]:
                        current_action = "Volume Up"
                        if volume_cooldown == 0:
                            pyautogui.press('volumeup')
                            volume_cooldown = VOLUME_CD
                        did_volume = True
                    elif fvec_wo_thumb == [True, False, False, False]:
                        current_action = "Volume Down"
                        if volume_cooldown == 0:
                            pyautogui.press('volumedown')
                            volume_cooldown = VOLUME_CD
                        did_volume = True

            # --- (B) マウス（鼻より右。※高さ条件なし） ---
            mouse_region_ok = True
            if nose_x is not None:
                mcp_hits_right = []
                for mcp_id in MCPs:
                    mcp_x = handLms.landmark[mcp_id].x * w
                    right_of_nose = (mcp_x > nose_x)
                    mcp_hits_right.append(right_of_nose)
                mouse_region_ok = any(mcp_hits_right)

            if (not did_volume) and mouse_region_ok:
                if fvec == [True, False, False, False, False]:
                    current_action = "Scroll Up"
                    if scroll_cooldown == 0:
                        pyautogui.scroll(SCROLL_STEP)
                        scroll_cooldown = SCROLL_CD
                elif fvec == [False, False, False, False, True]:
                    current_action = "Scroll Down"
                    if scroll_cooldown == 0:
                        pyautogui.scroll(-SCROLL_STEP)
                        scroll_cooldown = SCROLL_CD
                elif fvec == [False, True, False, False, False]:
                    current_action = "Click"
                    if gesture_state != "click":
                        pyautogui.click()
                        gesture_state = "click"
                elif fvec == [False, True, True, False, False]:
                    current_action = "Double Click"
                    if gesture_state != "double":
                        pyautogui.doubleClick()
                        gesture_state = "double"
                elif fvec == [True, True, False, False, False]:
                    current_action = "Right Click"
                    if gesture_state != "right":
                        pyautogui.rightClick()
                        gesture_state = "right"
                elif fvec == [False, False, False, False, False]:
                    current_action = "Dragging"
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                    gesture_state = "drag"
                else:
                    current_action = "Ready"
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    gesture_state = "Ready"

            # マウス領域外でドラッグし続けるのを防ぐ
            if (not mouse_region_ok) and dragging:
                pyautogui.mouseUp()
                dragging = False

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # クールダウン
    if scroll_cooldown > 0:
        scroll_cooldown -= 1
    if volume_cooldown > 0:
        volume_cooldown -= 1

    # デバッグ表示（音量用の高さ帯＆鼻xの縦線）
    if (ref_y is not None) and (face_h is not None):
        tol = int(face_h * NOSE_TOL_Y)
        cv2.line(frame, (0, int(ref_y - tol)), (w, int(ref_y - tol)), (120, 180, 120), 1)
        cv2.line(frame, (0, int(ref_y + tol)), (w, int(ref_y + tol)), (120, 180, 120), 1)
    if nose_x is not None:
        cv2.line(frame, (int(nose_x), 0), (int(nose_x), h), (80, 160, 220), 1)

    cv2.putText(frame, f"Action: {current_action}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    cv2.imshow("Gesture Control: Volume(L with height) / Mouse(R no height)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
