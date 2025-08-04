# (Goal)非接触でPC操作を実現する

## Task1: 左の瞳[landmark(473)]を使って、カーソルを移動(mediapipe)
### ・

## Task2: ハンドジェスチャー(片手どっちか)を取得し、マウスの機能を網羅する(mediapipe)
### ・クリック
### ・ダブルクリック
### ・右クリック
### ・アップスクロール
### ・ダウンスクロール
### ・ドラッグ＆ドロップ

## Task3: 耳を澄ますジェスチャーで音量UP & 耳を塞ぐジェスチャーで音量DOWN
### ・

## Task4: キーボード
### ・

## Task5: 人差し指の先端の移動で動画の操作
### ・

-> 顔の中心位置も追跡し、視線だけでなく**顔の姿勢（head pose）**も加味する
mediapipeなどを使うと、head pose estimationが可能

## 実行コード

### mediapipeを使用する場合
~~~bash!
pip install mediapipe
~~~

### pyautoguiを使用する場合
~~~bash!
pip install pyautogui
~~~

### geze trackingを使用する場合
~~~bash!
pip install cmake
pip install dlib
pip install opencv-python
pip install gaze-tracking
~~~

### Python_Gaze_Face_Trackerを使用する場合
(7/30 Yahagi edited)
~~~bash!
pip install opencv-python mediapipe pyautogui
~~~

* 補正:  スクリーンの左上->右上->左下->右下->真ん中の五段階で補正を行う。
* 目線＋顔の向きでマウスを動かすが、メインは顔の向きになりそう。顔を動かすとその角度に応じてマウスが動き、コード内の「alpha」の値でどれだけ早くカーソルが動くか（マウス感度）を変更することができる。
