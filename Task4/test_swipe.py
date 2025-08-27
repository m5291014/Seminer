import cv2
import mediapipe as mp
import pygame
import math
import pyautogui
from wordfreq import word_frequency

# --- Pygameの初期設定 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
KEY_COLOR = (255, 255, 255, 180)
TEXT_COLOR = (0, 0, 0)
PATH_COLOR = (255, 0, 0)
CANDIDATE_COLOR = (0, 0, 255)
HIGHLIGHT_COLOR = (255, 255, 0, 100) # ハイライト用の半透明黄色
FINGERTIP_COLOR = (0, 255, 0)
SWIPING_INDICATOR_COLOR = (0, 255, 255)

# --- MediaPipe ---
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# --- 辞書読み込み ---
def load_dictionary_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            words = [line.strip().lower() for line in file if line.strip().isalpha()]
            print(f"Loaded {len(words)} words from {filepath}")
            return words
    except FileNotFoundError:
        print(f"Dictionary file not found at '{filepath}'")
        return ["error", "dictionary", "not", "found"]

WORD_DICTIONARY = load_dictionary_from_file('words.txt')

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

# --- キーボード生成 ---
def create_keyboard_layout(is_symbol_layout, is_shift_on, font):
    keys, key_surfaces = {}, {}
    
    layout_qwerty = [
        [("q","q"), ("w","w"), ("e","e"), ("r","r"), ("t","t"), ("y","y"), ("u","u"), ("i","i"), ("o","o"), ("p","p")],
        [("a","a"), ("s","s"), ("d","d"), ("f","f"), ("g","g"), ("h","h"), ("j","j"), ("k","k"), ("l","l")],
        [("z","z"), ("x","x"), ("c","c"), ("v","v"), ("b","b"), ("n","n"), ("m","m")]
    ]
    
    layout_symbols = [
        [("1","1"),("2","2"),("3","3"),("4","4"),("5","5"),("6","6"),("7","7"),("8","8"),("9","9"),("0","0")],
        [("!","!"),("\"","@"),("#","#"),("$","$"),("%","%"),("&","^"),("\'","&"),("(","*"),(")","(")],
        [("-","-"),("=","_"),("^","^"),("~","~"),("¥","yen"),("|","|"),("@","["),("`","`")],
        [("[","]"),("]","\\"),("{","{"),("}","|"),(":","\'"),("+","add"),(";", ";"),("*","\"")],
        [(",",","),(".", "."),("<","<"),(">",">"),("?","?"),("/","/"),("_",'_')]
    ]
    
    layout = layout_symbols if is_symbol_layout else layout_qwerty
    key_size, padding, keyboard_y_offset = 60, 8, 120

    for i, row in enumerate(layout):
        start_x = (SCREEN_WIDTH - (len(row)*(key_size+padding)-padding))/2
        y = padding + i*(key_size+padding) + keyboard_y_offset
        for j, (label, value) in enumerate(row):
            x = start_x + j*(key_size+padding)
            if is_shift_on and not is_symbol_layout:
                label, value = label.upper(), value.upper()
            keys[(label,value)] = pygame.Rect(x, y, key_size, key_size)
            surf = pygame.Surface((key_size,key_size), pygame.SRCALPHA)
            surf.fill(KEY_COLOR)
            text_surf = font.render(label, True, TEXT_COLOR)
            text_rect = text_surf.get_rect(center=(key_size/2, key_size/2))
            surf.blit(text_surf, text_rect)
            key_surfaces[(label,value)] = surf
    
    last_row_y = list(keys.values())[-1].top
    special_keys_y = last_row_y + key_size + padding
    shift_width, space_width, backspace_width, switch_width = key_size*2, key_size*7, key_size*2, key_size*2
    start_x_special = (SCREEN_WIDTH - (shift_width+space_width+backspace_width+switch_width+padding*3))/2

    keys[("Shift","shift")] = pygame.Rect(start_x_special, special_keys_y, shift_width, key_size)
    keys[("Space"," ")] = pygame.Rect(keys[("Shift","shift")].right+padding, special_keys_y, space_width, key_size)
    keys[("BS","backspace")] = pygame.Rect(keys[("Space"," ")].right+padding, special_keys_y, backspace_width, key_size)
    keys[("SW","switch")] = pygame.Rect(keys[("BS","backspace")].right+padding, special_keys_y, switch_width, key_size)

    for (label,value), rect in keys.items():
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        surf.fill(KEY_COLOR)
        text_surf = font.render(label, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=(rect.width/2, rect.height/2))
        surf.blit(text_surf, text_rect)
        key_surfaces[(label,value)] = surf

    return keys, key_surfaces

# --- メイン ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Gesture Keyboard")
    font_key = pygame.font.Font(None, 32)
    font_ui = pygame.font.Font(None, 36)

    is_shift_on = False
    is_symbol_layout = False
    keys, key_surfaces = create_keyboard_layout(is_symbol_layout, is_shift_on, font_key)

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        is_swiping = False
        swipe_path, swiped_keys, candidates, candidate_rects = [], [], [], []
        highlighted_candidate_index = -1
        v_sign_frame_count, fist_frame_count = 0, 0
        GESTURE_CONFIRMATION_FRAMES = 3

        running = True
        while running and cap.isOpened():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            success, image = cap.read()
            if not success: continue
            image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            left_hand_landmarks, right_hand_landmarks = None, None
            fingertip_pos = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    if label=="Left": left_hand_landmarks=landmarks
                    else: right_hand_landmarks=landmarks

            if right_hand_landmarks:
                tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                ix, iy = int(tip.x*SCREEN_WIDTH), int(tip.y*SCREEN_HEIGHT)
                fingertip_pos = (ix, iy)

            # --- ジェスチャー ---
            v_sign_frame_count = v_sign_frame_count+1 if is_v_sign(left_hand_landmarks) else 0
            fist_frame_count = fist_frame_count+1 if is_fist(left_hand_landmarks) else 0

            # --- スワイプ開始/終了 ---
            if v_sign_frame_count==GESTURE_CONFIRMATION_FRAMES:
                is_swiping = not is_swiping
                if is_swiping: swipe_path, swiped_keys, candidates, candidate_rects = [fingertip_pos] if fingertip_pos else [], [], [], []
                else:
                    if len(swiped_keys)==1: pyautogui.write(swiped_keys[0])
                    elif len(swiped_keys)>1:
                        found_words=[w for w in WORD_DICTIONARY if w.startswith(swiped_keys[0]) and w.endswith(swiped_keys[-1])]
                        if found_words:
                            sorted_candidates=sorted(found_words,key=lambda w: word_frequency(w,'en'),reverse=True)
                            candidates = sorted_candidates[:5]

            # --- キー入力 ---
            if fist_frame_count==GESTURE_CONFIRMATION_FRAMES:
                if highlighted_candidate_index!=-1:
                    pyautogui.write(candidates[highlighted_candidate_index]+' ')
                    candidates=[]
                elif fingertip_pos:
                    for (label,value), rect in keys.items():
                        if rect.collidepoint(fingertip_pos):
                            if value=='shift':
                                is_shift_on=not is_shift_on
                                keys,key_surfaces=create_keyboard_layout(is_symbol_layout,is_shift_on,font_key)
                            elif value=='switch':
                                is_symbol_layout=not is_symbol_layout
                                keys,key_surfaces=create_keyboard_layout(is_symbol_layout,is_shift_on,font_key)
                            elif value=='space': pyautogui.press('space')
                            elif value=='backspace': pyautogui.press('backspace')
                            else: pyautogui.press(value.lower() if not is_shift_on else value.upper())
                            break

            # --- スワイプ軌跡 ---
            if is_swiping and fingertip_pos:
                if not swipe_path or (math.hypot(fingertip_pos[0]-swipe_path[-1][0],fingertip_pos[1]-swipe_path[-1][1])>5):
                    swipe_path.append(fingertip_pos)
                for char, rect in keys.items():
                    if rect.collidepoint(fingertip_pos):
                        if not swiped_keys or swiped_keys[-1]!=char: swiped_keys.append(char)
                        break

            # --- 描画 ---
            screen.fill(KEY_COLOR)
            for char, rect in keys.items():
                screen.blit(key_surfaces[char], rect.topleft)
                pygame.draw.rect(screen, TEXT_COLOR, rect, 1)

            highlighted_candidate_index=-1
            if not is_swiping and candidates:
                candidate_rects=[]
                cand_y_start = list(keys.values())[-1].bottom + 40
                total_height = len(candidates) * 40

                # はみ出す場合は上にずらす
                if cand_y_start + total_height > SCREEN_HEIGHT:
                    cand_y_start = SCREEN_HEIGHT - total_height - 20
                    
                for i, word in enumerate(candidates):
                    cand_surf = font_ui.render(f"{i+1}. {word}", True, CANDIDATE_COLOR, (255,255,255))
                    cand_rect = cand_surf.get_rect(topright=(SCREEN_WIDTH - 40, cand_y_start+i*40))
                    candidate_rects.append(cand_rect)
                    if fingertip_pos and cand_rect.collidepoint(fingertip_pos):
                        highlighted_candidate_index=i
                    screen.blit(cand_surf, cand_rect)
            if highlighted_candidate_index!=-1:
                highlight_surf=pygame.Surface(candidate_rects[highlighted_candidate_index].size, pygame.SRCALPHA)
                highlight_surf.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surf, candidate_rects[highlighted_candidate_index].topleft)

            if len(swipe_path)>1: pygame.draw.lines(screen, PATH_COLOR, False, swipe_path, 4)
            if fingertip_pos: pygame.draw.circle(screen, FINGERTIP_COLOR, fingertip_pos, 10, 3)
            if is_swiping: pygame.draw.circle(screen, SWIPING_INDICATOR_COLOR, (30,40), 15)

            pygame.display.flip()

    cap.release()
    pygame.quit()

if __name__=='__main__':
    main()
