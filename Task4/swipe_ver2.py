# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import pygame
import math
import pyautogui
import Levenshtein
import os
from word_freq import zipf_frequency

# --- Pygameの初期設定 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
KEY_COLOR = (255, 255, 255, 240)
TEXT_COLOR = (0, 0, 0)
PATH_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (255, 255, 0, 100)
FINGERTIP_COLOR = (0, 255, 0)
SWIPING_INDICATOR_COLOR = (0, 255, 255)

# --- MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# -------------------------
# ローマ字マップ（主要な五十音） — JPモードで送信するため
# -------------------------
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

# =========================
# 英語用辞書（スワイプ候補用）
# =========================
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

# --- ジェスチャー判定（そのまま） ---
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
    elif mode == 'sym':
        layout = layout_symbols
    else:  # 'jp'
        layout = layout_gojuon_shift if shift_toggle else layout_gojuon_normal

    key_size, padding, keyboard_y_offset = 60, 8, 120
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

    keys[("Shift", "shift")] = pygame.Rect(start_x_special, special_keys_y, shift_w, key_size)
    keys[("Space", " ")] = pygame.Rect(keys[("Shift","shift")].right+padding, special_keys_y, space_w, key_size)
    keys[("Enter", "enter")] = pygame.Rect(keys[("Space"," ")].right+padding, special_keys_y, enter_w, key_size)
    keys[("BS", "backspace")] = pygame.Rect(keys[("Enter","enter")].right+padding, special_keys_y, bs_w, key_size)
    keys[("SW", "switch")] = pygame.Rect(keys[("BS","backspace")].right+padding, special_keys_y, sw_w, key_size)

    # サーフ作成（特殊キーも）
    for (label, value), rect in list(keys.items()):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill(KEY_COLOR)
        text_surf = font.render(label, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(rect.width/2, rect.height/2))
        surf.blit(text_surf, text_rect)
        key_surfaces[(label, value)] = surf

    return keys, key_surfaces

# --- メイン ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Gesture Keyboard (EN/JP/SYM)")
    # mac の環境に合わせてフォントパスを必要なら変えてください
    font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W1.ttc"
    font_key = pygame.font.Font(font_path, 32)
    font_ui = pygame.font.Font(font_path, 20)

    # 状態
    shift_toggle = False   # トグル式
    modes = ['en', 'jp', 'sym']  # 切替順
    mode_index = 0
    mode = modes[mode_index]
    keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)

    # Mediapipe カメラ
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        is_swiping = False
        swipe_path = []
        swiped_keys = []
        candidates_all = []
        candidate_page = 0
        candidate_rects = []
        highlighted_candidate_index = -1

        v_sign_frame_count = 0
        fist_frame_count = 0
        GESTURE_CONFIRMATION_FRAMES = 3

        running = True
        while running and cap.isOpened():
            # --- Pygame イベント処理（ウィンドウ閉じる等） ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # --- カメラキャプチャ & Hands 推定 ---
            success, image = cap.read()
            if not success:
                continue
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

            if right_hand_landmarks:
                tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ix, iy = int(tip.x * SCREEN_WIDTH), int(tip.y * SCREEN_HEIGHT)
                fingertip_pos = (ix, iy)

            # --- ジェスチャー判定（連続フレームで安定化） ---
            v_sign_frame_count = v_sign_frame_count + 1 if is_v_sign(left_hand_landmarks) else 0
            fist_frame_count = fist_frame_count + 1 if is_fist(left_hand_landmarks) else 0

            # --- Vサインでスワイプ開始/終了（※ENモードのみスワイプを許可） ---
            if v_sign_frame_count == GESTURE_CONFIRMATION_FRAMES:
                # toggle swiping only when EN mode; otherwise ignore
                if mode == 'en':
                    is_swiping = not is_swiping
                    if is_swiping:
                        swipe_path = [fingertip_pos] if fingertip_pos else []
                        swiped_keys = []
                        candidates_all = []
                        candidate_page = 0
                    else:
                        # スワイプ終了時の確定処理（ENはスワイプによる候補探索、JP/SYMは通常スキップ）
                        if len(swiped_keys) == 1:
                            label, value = swiped_keys[0]
                            # EN 単独
                            pyautogui.press(value.lower())
                        elif len(swiped_keys) > 1:
                            # EN スワイプ候補検索
                            seq_first = swiped_keys[0][0]
                            seq_last = swiped_keys[-1][0]
                            found_words = [w for w in WORD_DICTIONARY if w.startswith(seq_first) and w.endswith(seq_last)]
                            found_words = list(set(found_words))
                            if found_words:
                                sorted_candidates = sorted(found_words, key=lambda w: candidate_score_lev(w, swiped_keys), reverse=True)
                                candidates_all = sorted_candidates
                                candidate_page = 0
                else:
                    # JP/SYM のときはスワイプ機能は無効 — 念のため状態をクリア
                    is_swiping = False
                    swipe_path = []
                    swiped_keys = []

            # --- 左手グーで「確定／ボタン押下」 ---
            if fist_frame_count == GESTURE_CONFIRMATION_FRAMES:
                # 1) 候補ハイライト確定
                if highlighted_candidate_index != -1 and candidates_all:
                    word_index = candidate_page * 5 + highlighted_candidate_index
                    if word_index < len(candidates_all):
                        pyautogui.write(candidates_all[word_index] + ' ')
                        candidates_all = []
                # 2) ボタン押下判定（フィンガーチップ座標があることが必須）
                elif fingertip_pos:
                    for (label, value), rect in keys.items():
                        if rect.collidepoint(fingertip_pos):
                            # 無効セル保護
                            if value == "": break

                            # 特殊キー処理
                            if value == 'shift':
                                # Shift トグル（押すたびに切替）
                                shift_toggle = not shift_toggle
                                keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)
                            elif value == 'switch':
                                # モード切替 (EN -> JP -> SYM -> EN ...)
                                mode_index = (mode_index + 1) % len(modes)
                                mode = modes[mode_index]
                                # IME 切替ショートカットを送る（あなたの環境に合わせたショートカット）
                                if mode == 'jp':
                                    pyautogui.hotkey('ctrl', 'shift', 'j')   # カナ入力 (あなた設定)
                                elif mode == 'en':
                                    pyautogui.hotkey('ctrl', 'shift', ';')   # 英数入力 (あなた設定)
                                # SYM は OS IME はそのままにして内部レイアウトだけ切替
                                keys, key_surfaces = create_keyboard_layout(font_key, mode=mode, shift_toggle=shift_toggle)
                            elif value == ' ':
                                pyautogui.press('space')
                            elif value == 'backspace':
                                pyautogui.press('backspace')
                            elif value == 'enter':
                                pyautogui.press('return')
                            elif value=='cand_up':   # ★ 上ページ
                                if candidate_page>0: candidate_page-=1
                            elif value=='cand_down': # ★ 下ページ
                                if (candidate_page+1)*5 < len(candidates_all): candidate_page+=1
                            elif value=='up':
                                pyautogui.press('up')
                            elif value=='down':
                                pyautogui.press('down')
                            else:
                                # 通常キー（モードごとに扱いを変える）
                                if mode == 'en':
                                    # EN: 単独入力
                                    pyautogui.press(value.lower() if len(value) == 1 else value)
                                elif mode == 'jp':
                                    # JP: 単独入力はその仮名に対応するローマ字を送る（IME に任せる）
                                    rom = ROMAJI_MAP.get(label, "")
                                    if rom:
                                        pyautogui.write(rom)
                                else:  # sym
                                    # 記号はそのまま書き込む
                                    pyautogui.write(value)
                            break

            # --- スワイプ軌跡＆スワイプ中のキー追跡（ENモードのみ） ---
            if is_swiping and fingertip_pos and mode == 'en':
                if not swipe_path or (math.hypot(fingertip_pos[0]-swipe_path[-1][0], fingertip_pos[1]-swipe_path[-1][1]) > 5):
                    swipe_path.append(fingertip_pos)
                for char, rect in keys.items():
                    # ENモード中は英字レイアウトのみ反応したいのでラベルが1文字のもの（英字）に限定
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
            highlighted_candidate_index = -1
            if not is_swiping and candidates_all:
                candidate_rects = []
                cand_y_start = list(keys.values())[-1].bottom + 40
                total_height = 5 * 40

                if cand_y_start + total_height > SCREEN_HEIGHT:
                    cand_y_start = SCREEN_HEIGHT - total_height - 20

                page_candidates = candidates_all[candidate_page*5:(candidate_page+1)*5]
                for i, word in enumerate(page_candidates):
                    cand_surf = font_ui.render(f"{i+1}. {word}", True, CANDIDATE_COLOR, (255,255,255))
                    cand_rect = cand_surf.get_rect(topright=(SCREEN_WIDTH - 40, cand_y_start + i*40))
                    candidate_rects.append(cand_rect)
                    if fingertip_pos and cand_rect.collidepoint(fingertip_pos):
                        highlighted_candidate_index = i
                    screen.blit(cand_surf, cand_rect)

            if highlighted_candidate_index != -1 and candidate_rects:
                highlight_surf = pygame.Surface(candidate_rects[highlighted_candidate_index].size, pygame.SRCALPHA)
                highlight_surf.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surf, candidate_rects[highlighted_candidate_index].topleft)

            # ページ切替ボタン（内部候補用）
            if candidates_all:
                btn_up = pygame.Rect(SCREEN_WIDTH-120, 20, 40, 40)
                btn_down = pygame.Rect(SCREEN_WIDTH-60, 20, 40, 40)
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

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()
