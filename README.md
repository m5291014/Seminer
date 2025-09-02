## (Goal)非接触でPC操作を実現する

### Task1: 眉間[landmark(268)]を使って、カーソルを移動(mediapipe)
##### ・

### Task2: ハンドジェスチャー(片手どっちか)を取得し、マウスの機能を網羅する(mediapipe)
##### ・クリック
##### ・ダブルクリック
##### ・右クリック
##### ・アップスクロール
##### ・ダウンスクロール
##### ・ドラッグ＆ドロップ

### Task3: 耳を澄ますジェスチャーで音量UP & 耳を塞ぐジェスチャーで音量DOWN(mediapipe)
##### ・

### Task4: キーボード
##### ・



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
