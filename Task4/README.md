### Task4: キーボード

### swipe_ver1.py
・キーボード入力の操作を右手だけで行えるようにした。
・基本状態(パー)で、手のひらの中心をトラッキングしてポインタを移動
・グーにしている間はスワイプ入力で、グーを解除すると候補単語表示
・パーの状態から親指を手のひら側に折り込むとキー入力＆単語候補決定


### swipe_ver1.py
~~~bash!
pip install cv2
pip install mediapipe
pip install pygame
pip install numpy
pip install pyautogui
pip install word_frequency
~~~

こちらのコードでは、語彙リスト(words.txt)を用いて軌跡から文字の候補をいくつか持ってくるようになっているため、実行の際にはtest_swipe.pyと同じ階層にwords.txtがないと上手く機能しない。

### swipe_ver2.py
<img width="912" height="740" alt="image" src="https://github.com/user-attachments/assets/e317e998-8f5a-4b7c-867c-c908c87ed5a6" />
<img width="868" height="696" alt="image" src="https://github.com/user-attachments/assets/2f96afd2-5a58-4286-a81c-89e835618ce5" />
<img width="868" height="696" alt="image" src="https://github.com/user-attachments/assets/86379866-23eb-4bac-a05d-45a5ae24ab45" />



~~~bash!
pip install cv2
pip install mediapipe
pip install pygame
pip install numpy
pip install pyautogui
pip install zipf_frequency (もしかしたらword_freqという名前かも)
pip install Levenshtein
~~~

#### 機能の違い
1. swipe入力の英単語候補の決め方を変更した。 <br>
  * 今まで: freq(頻度)のみで候補決定 <br>
  * 今回 : freq + Levenshtein.distanceという新たな指標を追加した。 <br>
  * この指標は、ある２つの単語に対して、どれくらいの文字の変化を加えれば同じ単語になるかというものを距離として計算するものである。 <br>
  今回においては、word(辞書にある単語)とswiped_keys(スワイプ時に通った文字の軌跡)を入力として、距離を計算している。 <br>
  最終的に、freqとdistに重みをかけ、それを足し合わせたものをスコアとしてそれを元に候補を並び替えている。

2. freqのライブラリ変更。 <br>
  * 今まで: word_frequency, (0.0 ~ 1.0の指標) <br>
  * 今回 : zipf_frequency, (0 ~ 7の指標) <br>
  
  同じwordfreqライブラリからのインポートなので、機能的には違いはないが、Levenshteinが大きめの値になりがちなので、使い勝手が良いzipf_frequencyを使用した。
~~~bash!
(63~68行目)
# Levenshteinの編集距離を使って候補を出している。
def candidate_score_lev(word, swiped_keys):
    freq = zipf_frequency(word, 'en')
    seq = ''.join([k[0] if isinstance(k, tuple) else k for k in swiped_keys]).lower()
    dist = Levenshtein.distance(word, seq)
    return freq * 2 - dist * 0.5  # 距離が大きいとスコア下げる
~~~

3. 日本語かな入力の追加。<br>
  普通の50音順の配列に加え、Shiftキーによって小文字や濁点半濁点なども入力できるようになっている。予測変換については、現在使用しているものとライブラリの相性が良くなかったため、実装していない。日本語入力の時に画面上に表示される候補（デフォルトの）を上下矢印で選択して変換を行うようにした。

4. キーの編集。　<br>
  エンターキーの実装。 <br>
  記号キーの種類の削減。<br>
  
5. SW(swith)ボタンの改善。 <br>
  キー配列が３種類になったので、EN -> JP -> 記号 -> ... という順番でキーの種類が循環する。 <br>

#### 注意点
~~~bash!
(97~101行目)
# SYM 簡易レイアウト
layout_symbols = [
    [("1","1"),("2","2"),("3","3"),("4","4"),("5","5"),("6","6"),("7","7"),("8","8"),("9","9"),("0","0")],
    [(",",","),(".", "."),(":","\'"),(";", ";"),("?", "?"),("!","!"),("\"","@"),("'","&"),("(", "*"),(")", "(")]
]
~~~
* 上記は、記号の配列であり、("","")の左側は、配列に表示する記号、右側は入力するボタン。
* なぜかmacの環境では、記号のボタンの対応が同じではないらしく、pyautoguiに「"」を入力させようと思ったら、「@」を押すように指示しなければならないなどの問題がある。
* そこで、１つ１つが何に対応するかを確認してその対応を反映させたのが上記の配列であり、これはおそらくmac以外ではまた違う対応が存在すると思うので、ここでバグが発生するかも。

~~~bash!
(173~176行目)
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W1.ttc"
font_key = pygame.font.Font(font_path, 32)
font_ui = pygame.font.Font(font_path, 20)
~~~
* 上記は、pygame中のフォントを指定する箇所であり、これもOSによってフォントのpathが異なる。
* 元々は、pygame.font.Font(none, 40)のような指定だったが、これだと日本語の表示が文字化けしてしまうようだったので、直接フォントがある場所を指定している。
* windowsの場合は、font_pathの中を自身の対応する日本語のフォントのpathに置き換える必要がある。
