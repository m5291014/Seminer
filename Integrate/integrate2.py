import pygame
import sys
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math
import Levenshtein
from wordfreq import zipf_frequency

isDebug = False  # Trueなら、目線移動の際、カメラの情報が描画される。

# --- Pygameの初期設定 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
KEY_COLOR = (255, 255, 255, 240)
TEXT_COLOR = (0, 0, 0)
PATH_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (255, 255, 0, 100)
FINGERTIP_COLOR = (0, 255, 0)
SWIPING_INDICATOR_COLOR = (0, 255, 255)

# --- Pygame 初期化 ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Mode Switcher")
#mac
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W1.ttc"
#windows
#font_path = "C:/Windows/Fonts/meiryo.ttc" 
font_key = pygame.font.Font(font_path, 32)
font_ui = pygame.font.Font(font_path, 20)
clock = pygame.time.Clock()

# --- モード管理 ---
mode_operate = "mouse"  # 初期モード

# --- 切替ボタンの矩形 ---
switch_button = pygame.Rect(SCREEN_WIDTH/2, 20, 120, 50)
switch_cooldown = 10  # 👊で切り替える際に、クールダウンがないと、切り替えが何回も連続で起きてしまう。

# --- MediaPipe 初期化 ---
screen_w, screen_h = pyautogui.size()
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# --- 視線キャリブレーション関連 ---
calibration_points = []
gaze_min = [1.0, 1.0]
gaze_max = [0.0, 0.0]
alpha = 0.3
calibrated = False
b_is_fist = False
fist_cooldown = 0  # キャリブレーションの際にグーのクールダウン

guide_texts = [
    "Look at the TOP LEFT corner and press SPACE or grip your hand",
    "Look at the TOP RIGHT corner and press SPACE or grip your hand",
    "Look at the BOTTOM LEFT corner and press SPACE or grip your hand",
    "Look at the BOTTOM RIGHT corner and press SPACE or grip your hand",
    "Look at the CENTER and press SPACE or grip your hand"
]

# 終了処理のピースサイン、持続カウントとその閾値
peace_counter = 0
PEACE_THRESHOLD = 10

# --- ハンドジェスチャー関連 ---
gesture_state = "Ready"
current_action = "None"
scroll_cooldown = 0
dragging = False
tips_ids = [4, 8, 12, 16, 20]

# --- 音量関連 ----
VOLUME_CD = 3          
NOSE_TOL_Y = 0.12       
USE_EAR_HEIGHT_IF_AVAILABLE = True
SCROLL_CD = 3         
SCROLL_STEP = 100        
try:
    volume_cooldown
except NameError:
    volume_cooldown = 0

try:
    mp_fd
    face_det
except NameError:
    mp_fd = mp.solutions.face_detection
    face_det = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.6)



# キーボード関連
shift_toggle = False
modes = ['en', 'jp', 'sym']
mode_index = 0

is_swiping = False
swipe_path = []
swiped_keys = []
candidates_all = []
candidate_page = 0
candidate_rects = []
highlighted_candidate_index = -1

thumb_click_frame_count = 0  # ★変更: 左手グーの代わりに右手親指クリック判定用
GESTURE_CONFIRMATION_FRAMES = 3

# --- 追加: OpenCVウィンドウ状態フラグ ---
mouse_window_created = False

# --- OpenCVウィンドウの安全な表示/破棄ヘルパー ---
def set_mouse_window_visible(visible, frame=None):
    """Mouse Mode ウィンドウの表示/非表示を安全に切り替える"""
    global mouse_window_created
    if visible:
        if not mouse_window_created:
            try:
                cv2.namedWindow("Mouse Mode", cv2.WINDOW_NORMAL)
            except cv2.error:
                pass
            mouse_window_created = True
        if frame is not None:
            try:
                cv2.imshow("Mouse Mode", frame)
            except cv2.error:
                pass
    else:
        if mouse_window_created:
            try:
                cv2.destroyWindow("Mouse Mode")
            except cv2.error:
                pass
            mouse_window_created = False

# ローマ字マップ
ROMAJI_MAP = {
    "あ":"a","い":"i","う":"u","え":"e","お":"o",
    "か":"ka","き":"ki","く":"ku","け":"ke","こ":"ko",
    "さ":"sa","し":"shi","す":"su","せ":"se","そ":"so",
    "た":"ta","ち":"chi","つ":"tsu","て":"te","と":"to",
    "な":"na","に":"ni","ぬ":"nu","ね":"ne","の":"no",
    "は":"ha","ひ":"hi","ふ":"fu","へ":"he","ほ":"ho",
    "ま":"ma","み":"mi","む":"mu","め":"me","も":"mo",
    "や":"ya","ゆ":"yu","よ":"yo",
    "ら":"ra","り":"ri","る":"ru","れ":"re","ろ":"ro",
    "わ":"wa","を":"wo","ん":"nn",
    "ぁ":"xa","ぃ":"xi","ぅ":"xu","ぇ":"xe","ぉ":"xo",
    "ゃ":"xya","ゅ":"xyu","ょ":"xyo","っ":"xtsu",
    "が":"ga","ぎ":"gi","ぐ":"gu","げ":"ge","ご":"go",
    "ざ":"za","じ":"zi","ず":"zu","ぜ":"ze","ぞ":"zo",
    "だ":"da","ぢ":"di","づ":"du","で":"de","ど":"do",
    "ば":"ba","び":"bi","ぶ":"bu","べ":"be","ぼ":"bo",
    "ぱ":"pa","ぴ":"pi","ぷ":"pu","ぺ":"pe","ぽ":"po",
    "ー":"-","。":".","、":","
}

def detect_fingers_up(landmarks):
    fingers = []
    if landmarks.landmark[tips_ids[0]].x < landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(True)
    else:
        fingers.append(False)
    for i in range(1, 5):
        fingers.append(
            landmarks.landmark[tips_ids[i]].y < landmarks.landmark[tips_ids[i] - 2].y
        )
    return fingers

# 英語用辞書
def load_dictionary_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            words = [line.strip().lower() for line in file if line.strip().isalpha()]
            print(f"[DICT] Loaded {len(words)} words from {filepath}")
            return words
    except FileNotFoundError:
        print(f"[DICT] file not found: {filepath} — using small fallback")
        return ["test", "example", "hello", "world"]

WORD_DICTIONARY = load_dictionary_from_file('words.txt')

def candidate_score_lev(word, swiped_keys):
    freq = zipf_frequency(word, 'en')
    seq = ''.join([k[0] if isinstance(k, tuple) else k for k in swiped_keys]).lower()
    dist = Levenshtein.distance(word, seq)
    return freq * 2 - dist * 0.5  # 距離が大きいとスコア下げる

# --- ジェスチャー判定 ---
def is_v_sign(hand_landmarks):
    if not hand_landmarks: return False
    return (hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y and
            hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y and
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
            hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y)

def is_fist(hand_landmarks):
    if not hand_landmarks: return False
    return (hand_landmarks.landmark[8].y > hand_landmarks.landmark[6].y and
            hand_landmarks.landmark[12].y > hand_landmarks.landmark[10].y and
            hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y and
            hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y)



# --- ★改良版: 手のひら中心を計算（右半分のみ使用＆スケーリング） ---
def get_palm_center(hand_landmarks, screen_width, screen_height,
                    x_min=0.5, x_max=1.0):  # ← ROIを指定（右半分なら0.5〜1.0）
    if not hand_landmarks:
        return None
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    # ROI外なら無効
    if not (x_min <= cx <= x_max):
        return None
    # ROI内座標を0〜1に正規化
    cx_roi = (cx - x_min) / (x_max - x_min)
    cy_roi = cy  # y方向は全域を使用（必要なら制限可）

    return (int(cx_roi * screen_width), int(cy_roi * screen_height))

def is_thumb_click(hand_landmarks):
    if not hand_landmarks:
        return False
    lm = hand_landmarks.landmark

    # 手のひらの輪郭を構成するランドマーク（0:手首, 1:親指付け根, 5:人差し指付け根,
    # 9:中指付け根, 13:薬指付け根, 17:小指付け根）
    palm_points = [(lm[i].x, lm[i].y) for i in [0, 1, 5, 9, 13, 17]]

    # 親指先端
    thumb_tip = (lm[4].x, lm[4].y)

    # 多角形に変換（画像座標系に合わせるため配列化）
    palm_polygon = np.array(palm_points, dtype=np.float32)

    # 親指先端がポリゴンの内側にあるかを判定
    result = cv2.pointPolygonTest(palm_polygon, thumb_tip, False)

    return result >= 0  # 内側(>0)または辺上(=0)ならTrue

# --- キーボード生成: モード（en/jp/sym）と shift_toggle に応じたレイアウトを返す ---
def create_keyboard_layout(font, mode='en', shift_toggle=False):
    keys = {}
    key_surfaces = {}

    # EN レイアウト（QWERTY）
    layout_qwerty = [
        [("q","q"), ("w","w"), ("e","e"), ("r","r"), ("t","t"), ("y","y"), ("u","u"), ("i","i"), ("o","o"), ("p","p")],
        [("a","a"), ("s","s"), ("d","d"), ("f","f"), ("g","g"), ("h","h"), ("j","j"), ("k","k"), ("l","l")],
        [("z","z"), ("x","x"), ("c","c"), ("v","v"), ("b","b"), ("n","n"), ("m","m")]
    ]

    # SYM 簡易レイアウト
    layout_symbols = [
        [("1","1"),("2","2"),("3","3"),("4","4"),("5","5"),("6","6"),("7","7"),("8","8"),("9","9"),("0","0")],
        [(",",","),(".", "."),(":","\'"),(";", ";"),("?", "?"),("!","!"),("\"","@"),("'","&"),("(", "*"),(")", "(")]
    ]

    # JP 通常レイアウト（横並びの五十音）
    layout_gojuon_normal = [
        [("あ","あ"),("か","か"),("さ","さ"),("た","た"),("な","な"),("は","は"),("ま","ま"),("や","や"),("ら","ら"),("わ","わ"),("、","、")],
        [("い","い"),("き","き"),("し","し"),("ち","ち"),("に","に"),("ひ","ひ"),("み","み"),("",""),("り","り"),("",""),("。","。")],
        [("う","う"),("く","く"),("す","す"),("つ","つ"),("ぬ","ぬ"),("ふ","ふ"),("む","む"),("ゆ","ゆ"),("る","る"),("を","を"),("ー","ー")],
        [("え","え"),("け","け"),("せ","せ"),("て","て"),("ね","ね"),("へ","へ"),("め","め"),("",""),("れ","れ"),("",""),("↑","up")],
        [("お","お"),("こ","こ"),("そ","そ"),("と","と"),("の","の"),("ほ","ほ"),("も","も"),("よ","よ"),("ろ","ろ"),("ん","ん"),("↓","down")],
    ]

    # JP Shift トグル時（小文字・濁点・半濁点を含めた置換レイアウト）
    layout_gojuon_shift = [
        [("ぁ","ぁ"),("が","が"),("ざ","ざ"),("だ","だ"),("な","な"),("ば","ば"),("ぱ","ぱ"),("ゃ","ゃ"),("",""),("",""),("、","、")],
        [("ぃ","ぃ"),("ぎ","ぎ"),("じ","じ"),("ぢ","ぢ"),("に","に"),("び","び"),("ぴ","ぴ"),("",""),("",""),("",""),("。","。")],
        [("ぅ","ぅ"),("ぐ","ぐ"),("ず","ず"),("づ","づ"),("ぬ","ぬ"),("ぶ","ぶ"),("ぷ","ぷ"),("ゅ","ゅ"),("",""),("",""),("ー","ー")],
        [("ぇ","ぇ"),("げ","げ"),("ぜ","ぜ"),("で","で"),("ね","ね"),("べ","べ"),("ぺ","ぺ"),("",""),("",""),("",""),("↑","up")],
        [("ぉ","ぉ"),("ご","ご"),("ぞ","ぞ"),("ど","ど"),("の","の"),("ぼ","ぼ"),("ぽ","ぽ"),("ょ","ょ"),("",""),("っ","っ"),("↓","down")],
    ]

    if mode == 'en':
        layout = layout_qwerty
        keyboard_y_offset = 80
    elif mode == 'sym':
        layout = layout_symbols
        keyboard_y_offset = 120
    else:  # 'jp'
        layout = layout_gojuon_shift if shift_toggle else layout_gojuon_normal
        keyboard_y_offset = 120

    key_size, padding = 60, 8
    for i, row in enumerate(layout):
        start_x = (SCREEN_WIDTH - (len(row)*(key_size+padding)-padding))/2
        y = padding + i*(key_size+padding) + keyboard_y_offset
        for j, (label, value) in enumerate(row):
            if label == "" and value == "":  # 空セルスキップ
                continue
            x = start_x + j*(key_size+padding)
            rect = pygame.Rect(x, y, key_size, key_size)
            keys[(label, value)] = rect
            surf = pygame.Surface((key_size, key_size), pygame.SRCALPHA)
            surf.fill(KEY_COLOR)
            text_surf = font.render(label, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(key_size/2, key_size/2))
            surf.blit(text_surf, text_rect)
            key_surfaces[(label, value)] = surf

    # 特殊キー（Shiftトグル / Space / Enter / Backspace / Switch）
    last_row_y = list(keys.values())[-1].top
    special_keys_y = last_row_y + key_size + padding
    shift_w, space_w, enter_w, bs_w, sw_w = 90, 200, 90, 120, 90
    start_x_special = (SCREEN_WIDTH - (shift_w + space_w + enter_w + bs_w + sw_w + padding*4)) / 2
    switch_rect = pygame.Rect(SCREEN_WIDTH/2, 20, 120, 50)

    keys[("Shift", "shift")] = pygame.Rect(start_x_special, special_keys_y, shift_w, key_size)
    keys[("Space", " ")] = pygame.Rect(keys[("Shift","shift")].right+padding, special_keys_y, space_w, key_size)
    keys[("Enter", "enter")] = pygame.Rect(keys[("Space"," ")].right+padding, special_keys_y, enter_w, key_size)
    keys[("BS", "backspace")] = pygame.Rect(keys[("Enter","enter")].right+padding, special_keys_y, bs_w, key_size)
    keys[("SW", "switch")] = pygame.Rect(keys[("BS","backspace")].right+padding, special_keys_y, sw_w, key_size)
    keys[("SWITCH", "switch_mode")] = switch_rect

    # サーフ作成（特殊キーも）
    for (label, value), rect in list(keys.items()):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill(KEY_COLOR)
        text_surf = font.render(label, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(rect.width/2, rect.height/2))
        surf.blit(text_surf, text_rect)
        key_surfaces[(label, value)] = surf

    return keys, key_surfaces

mode = modes[mode_index]
keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)

# --- マウスモード処理 ---
def mouse_mode():
    """
    視線キャリブ（FaceMesh, 5点）＋ 視線カーソル移動 を維持しつつ、
    鼻より右=マウス操作、鼻より左かつ高さ帯=音量ジェスチャ を追加。
    ウィンドウは set_mouse_window_visible で安全制御。
    左手グー=keyboard切替、左手Vサイン=終了判定は従来どおり。
    """
    global calibration_points, gaze_min, gaze_max, alpha, calibrated
    global b_is_fist, fist_cooldown, screen_w, screen_h
    global gesture_state, dragging, scroll_cooldown, volume_cooldown
    global mode_operate, switch_cooldown, peace_counter

    if switch_cooldown > 0:
        switch_cooldown -= 1

    ret, frame = cap.read()
    if not ret:
        return True

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- 検出（FaceMesh: 視線/キャリブ, FaceDetection: 鼻/耳, Hands: ジェスチャ） ----
    fm = face_mesh.process(rgb)          # 既存の mp_face_mesh.FaceMesh
    fd = face_det.process(rgb)           # 追加した FaceDetection
    hr = hands.process(rgb)              # 既存の Hands

    current_action = "None"

    # =========================
    # 1) 視線キャリブ & 視線カーソル
    # =========================
    if fm.multi_face_landmarks:
        face_landmarks = fm.multi_face_landmarks[0]
        gaze_x = face_landmarks.landmark[168].x
        gaze_y = face_landmarks.landmark[168].y

        if not calibrated:
            idx = len(calibration_points)
            if idx < 5:
                guide_text = guide_texts[idx]
                cv2.putText(frame, guide_text, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # SPACE or グー でサンプル取得
            key = cv2.waitKey(1) & 0xFF
            if key == 32 or b_is_fist:
                calibration_points.append((gaze_x, gaze_y))
                b_is_fist = False
                if len(calibration_points) == 5:
                    gaze_min[0] = min(p[0] for p in calibration_points)
                    gaze_max[0] = max(p[0] for p in calibration_points)
                    gaze_min[1] = min(p[1] for p in calibration_points)
                    gaze_max[1] = max(p[1] for p in calibration_points)
                    calibrated = True
                    cv2.waitKey(200)

            # 未キャリブ中は必ずプレビューを表示してこのフレームを終了
            set_mouse_window_visible(True, frame)
            return True

        else:
            # 視線 → 画面座標へ、スムージング移動
            norm_x = np.clip((gaze_x - gaze_min[0]) / (gaze_max[0] - gaze_min[0]), 0, 1)
            norm_y = np.clip((gaze_y - gaze_min[1]) / (gaze_max[1] - gaze_min[1]), 0, 1)
            screen_x = screen_w * norm_x
            screen_y = screen_h * norm_y
            px, py = pyautogui.position()
            smooth_x = px * (1 - alpha) + screen_x * alpha
            smooth_y = py * (1 - alpha) + screen_y * alpha
            pyautogui.moveTo(smooth_x, smooth_y)
            cv2.circle(frame, (int(gaze_x * w), int(gaze_y * h)), 5, (0, 255, 0), -1)

    # =========================
    # 2) 鼻x・参照y（耳優先）を取得（音量帯と左右ゲート用）
    # =========================
    ref_y = None
    face_h = None
    nose_x = None

    if fd.detections:
        det = fd.detections[0]
        rbb = det.location_data.relative_bounding_box
        y0 = int(rbb.ymin * h); y1 = int((rbb.ymin + rbb.height) * h)
        face_h = max(1, y1 - y0)

        kps = det.location_data.relative_keypoints
        if kps:
            nose = kps[2]
            nose_x = nose.x * w
            use_ear = False
            if USE_EAR_HEIGHT_IF_AVAILABLE and len(kps) >= 6:
                ry = kps[4].y * h if kps[4] is not None else None
                ly = kps[5].y * h if kps[5] is not None else None
                if (ry is not None) and (ly is not None):
                    ref_y = (ry + ly) / 2.0
                    use_ear = True
            if not use_ear:
                ref_y = nose.y * h

    # =========================
    # 3) ハンド（左：切替/終了、右：操作。片手ならその手）
    # =========================
    left_hand = None
    right_hand = None
    if hr.multi_hand_landmarks and hr.multi_handedness:
        for handLms, handed in zip(hr.multi_hand_landmarks, hr.multi_handedness):
            label = handed.classification[0].label  # "Left" / "Right"
            if label == "Left":
                left_hand = handLms
            else:
                right_hand = handLms

    # 左手のショートカット
    if left_hand is not None:
        if is_fist(left_hand) and switch_cooldown == 0:
            mode_operate = "keyboard"
            switch_cooldown = 10
        if is_v_sign(left_hand):
            peace_counter += 1
            if peace_counter == PEACE_THRESHOLD:
                return False
        else:
            peace_counter = 0


    # 右手が優先。無ければ左手でも操作可
    target_hand = right_hand or left_hand

    if target_hand is not None:
        fvec = detect_fingers_up(target_hand)       # [thumb, index, middle, ring, pinky]
        fvec_wo_thumb = fvec[1:]

        # 高さ帯（音量用）
        tol = None
        if (ref_y is not None) and (face_h is not None):
            tol = face_h * NOSE_TOL_Y
            top = ref_y - tol
            bot = ref_y + tol

        # 鼻の左右ゲート
        mouse_region_ok = True
        left_region_ok = False
        if nose_x is not None:
            MCPs = [5, 9, 13, 17]
            xs = [target_hand.landmark[m].x * w for m in MCPs]
            ys = [target_hand.landmark[m].y * h for m in MCPs]
            mouse_region_ok = any(x > nose_x for x in xs)  # 鼻より右 → マウス
            if tol is not None:
                left_region_ok = any((x < nose_x) and (top <= y <= bot) for x, y in zip(xs, ys))

        # 音量：鼻より左 ＆ 高さ帯内
        did_volume = False
        if left_region_ok and tol is not None:
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

        # マウス：鼻より右（高さ条件なし）
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

        # 鼻右でなくなったらドラッグは解除
        if (not mouse_region_ok) and dragging:
            pyautogui.mouseUp()
            dragging = False

        # 骨格描画
        mp_draw.draw_landmarks(frame, target_hand, mp_hands.HAND_CONNECTIONS)

    # クールダウン
    if scroll_cooldown > 0:
        scroll_cooldown -= 1
    if volume_cooldown > 0:
        volume_cooldown -= 1

    cv2.putText(frame, f"Action: {current_action}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

    # 表示制御：キャリブは上で表示済み・return済み。ここはキャリブ後の通常フレーム。
    if isDebug:
        set_mouse_window_visible(True, frame)
    else:
        set_mouse_window_visible(False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False
    return True

# --- キーボードモード処理 ---
def keyboard_mode(screen):
    global shift_toggle, mode_index, mode, keys, key_surfaces
    global candidates_all, candidate_page, highlighted_candidate_index
    global swipe_path, is_swiping, mode_operate, switch_cooldown
    global swiped_keys, peace_counter, thumb_click_frame_count

    if switch_cooldown > 0:
        switch_cooldown -= 1

    # --- カメラキャプチャ & Hands 推定 ---
    success, image = cap.read()
    if not success:
        return

    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    left_hand_landmarks = None
    right_hand_landmarks = None
    fingertip_pos = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            if label == "Left":
                left_hand_landmarks = landmarks
            else:
                right_hand_landmarks = landmarks

    if left_hand_landmarks:
        if is_fist(left_hand_landmarks) and switch_cooldown == 0:
            mode_operate = "mouse"
            switch_cooldown = 10
        if is_v_sign(left_hand_landmarks):
            peace_counter += 1
            if peace_counter == PEACE_THRESHOLD:
                cap.release()
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
                pygame.quit()
                sys.exit()
        else:
            peace_counter = 0

    # --- ★変更: 人差し指先端ではなく手のひら中心をトラッキング ---
    if right_hand_landmarks:
        fingertip_pos = get_palm_center(
            right_hand_landmarks, SCREEN_WIDTH, SCREEN_HEIGHT,
            x_min=0.6, x_max=1.0   # 右40%だけを使用
        )

    # --- ★変更: 右手グーの間だけスワイプON ---
    if right_hand_landmarks and is_fist(right_hand_landmarks):
        if not is_swiping:
            is_swiping = True
            swipe_path = [fingertip_pos] if fingertip_pos else []
            swiped_keys = []
            candidates_all = []
            candidate_page = 0
    else:
        if is_swiping:
            # スワイプ終了時の確定処理（ENモードのみ）
            if mode == 'en':
                if len(swiped_keys) == 1:
                    label, value = swiped_keys[0]
                    pyautogui.press(value.lower())
                elif len(swiped_keys) > 1:
                    seq_first = swiped_keys[0][0]
                    seq_last = swiped_keys[-1][0]
                    found_words = [w for w in WORD_DICTIONARY if w.startswith(seq_first) and w.endswith(seq_last)]
                    found_words = list(set(found_words))
                    if found_words:
                        sorted_candidates = sorted(found_words, key=lambda w: candidate_score_lev(w, swiped_keys), reverse=True)
                        candidates_all = sorted_candidates
                        candidate_page = 0
            is_swiping = False

    # --- ★変更: 親指クリックで「確定／ボタン押下」 ---
    thumb_click_frame_count = thumb_click_frame_count + 1 if is_thumb_click(right_hand_landmarks) else 0
    if thumb_click_frame_count == GESTURE_CONFIRMATION_FRAMES:
        if highlighted_candidate_index != -1 and candidates_all:
            word_index = candidate_page * 5 + highlighted_candidate_index
            if word_index < len(candidates_all):
                pyautogui.write(candidates_all[word_index] + ' ')
                candidates_all = []
        elif fingertip_pos:
            for (label, value), rect in keys.items():
                if rect.collidepoint(fingertip_pos):
                    if value == "": continue
                    if value == 'shift':
                        shift_toggle = not shift_toggle
                        keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)
                    elif value == 'switch':
                        mode_index = (mode_index + 1) % len(modes)
                        mode = modes[mode_index]
                        if mode == 'jp':
                            pyautogui.hotkey('ctrl', 'shift', 'j')
                        elif mode == 'en':
                            pyautogui.hotkey('ctrl', 'shift', ';')
                        keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)
                    elif value == ' ':
                        pyautogui.press('space')
                    elif value == 'backspace':
                        pyautogui.press('backspace')
                    elif value == 'enter':
                        pyautogui.press('return')
                    elif value == 'cand_up':
                        if candidate_page > 0: candidate_page -= 1
                        print("cand_up")
                    elif value == 'cand_down':
                        if (candidate_page+1)*5 < len(candidates_all): candidate_page += 1
                        print("cand_down")
                    elif value == 'up':
                        pyautogui.press('up')
                    elif value == 'down':
                        pyautogui.press('down')
                    elif value == 'switch_mode':
                        mode_operate = "mouse"
                        # Mouseウィンドウはmouse_mode内で管理される
                        return
                    else:
                        if mode == 'en':
                            pyautogui.press(value.lower() if len(value) == 1 else value)
                        elif mode == 'jp':
                            rom = ROMAJI_MAP.get(label, "")
                            if rom:
                                pyautogui.write(rom)
                        else:
                            pyautogui.write(value)
                    break

    # --- スワイプ軌跡 ---
    if is_swiping and fingertip_pos and mode == 'en':
        if not swipe_path or (math.hypot(fingertip_pos[0]-swipe_path[-1][0], fingertip_pos[1]-swipe_path[-1][1]) > 5):
            swipe_path.append(fingertip_pos)
        for char, rect in keys.items():
            label, val = char
            if rect.collidepoint(fingertip_pos):
                if not swiped_keys or swiped_keys[-1] != char:
                    swiped_keys.append(char)
                break

    # --- 描画: キー・候補・軌跡・指先表示 ---
    screen.fill(KEY_COLOR)
    for char, rect in keys.items():
        if char in [("UP","cand_up"),("DN","cand_down")]:
            continue  # 特別扱いするのでスキップ
        screen.blit(key_surfaces[char], rect.topleft)
        pygame.draw.rect(screen, TEXT_COLOR, rect, 1)

    # 内部候補表示（ENのスワイプ候補など）
    global candidate_rects  # 再代入のため
    highlighted_candidate_index = -1
    if not is_swiping and candidates_all:
        candidate_rects = []
        page_candidates = candidates_all[candidate_page*5:(candidate_page+1)*5]

        # ★中央下部に表示するための位置計算
        total_height = len(page_candidates) * 38
        cand_y_start = SCREEN_HEIGHT - total_height - 30  # 下端から余白
        for i, word in enumerate(page_candidates):
            cand_surf = font_ui.render(f"{i+1}. {word}", True, CANDIDATE_COLOR, (255,255,255))

            # ★中央揃え
            cand_rect = cand_surf.get_rect(center=(SCREEN_WIDTH // 2, cand_y_start + i * 30))
            candidate_rects.append(cand_rect)

            # ハイライト判定
            if fingertip_pos and cand_rect.collidepoint(fingertip_pos):
                highlighted_candidate_index = i

            screen.blit(cand_surf, cand_rect)

    if highlighted_candidate_index != -1 and candidate_rects:
        highlight_surf = pygame.Surface(candidate_rects[highlighted_candidate_index].size, pygame.SRCALPHA)
        highlight_surf.fill(HIGHLIGHT_COLOR)
        screen.blit(highlight_surf, candidate_rects[highlighted_candidate_index].topleft)

    # ページ切替ボタン（内部候補用）
    if candidates_all:
        btn_up = pygame.Rect(SCREEN_WIDTH-300, 370, 60, 60)
        btn_down = pygame.Rect(SCREEN_WIDTH-300, 440, 60, 60)
        keys[("UP","cand_up")] = btn_up
        keys[("DN","cand_down")] = btn_down
        for (label, value), rect in [(("UP","cand_up"), btn_up), (("DN","cand_down"), btn_down)]:
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            surf.fill((220,220,220,200))
            text_surf = font_ui.render(label, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(rect.width/2, rect.height/2))
            surf.blit(text_surf, text_rect)
            screen.blit(surf, rect.topleft)
            pygame.draw.rect(screen, TEXT_COLOR, rect, 2)

    if len(swipe_path) > 1:
        pygame.draw.lines(screen, PATH_COLOR, False, swipe_path, 4)
    if fingertip_pos:
        pygame.draw.circle(screen, FINGERTIP_COLOR, fingertip_pos, 8, 3)
    if is_swiping:
        pygame.draw.circle(screen, SWIPING_INDICATOR_COLOR, (30,40), 15)

    # モード表示（左上）
    mode_label_surf = font_ui.render(f"MODE: {mode.upper()}  SHIFT:{'ON' if shift_toggle else 'OFF'}", True, (0,0,0))
    screen.blit(mode_label_surf, (10, 10))

# --- メインループ ---
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if switch_button.collidepoint(event.pos):
                mode_operate = "keyboard" if mode_operate == "mouse" else "mouse"
                print(f"Switched to {mode_operate} mode")

    if mode_operate == "mouse":
        if not mouse_mode():  # 視線＋ハンドジェスチャー制御
            running = False
    elif mode_operate == "keyboard":
        keyboard_mode(screen)

    # 切替ボタンの描画
    pygame.draw.rect(screen, (200, 0, 0), switch_button)
    btn_text = font_key.render("Switch", True, (255, 255, 255))
    screen.blit(btn_text, (switch_button.x + 10, switch_button.y + 5))

    pygame.display.flip()
    clock.tick(30)

# --- 終了処理 ---
cap.release()
try:
    cv2.destroyAllWindows()
except cv2.error:
    pass
pygame.quit()
sys.exit()
