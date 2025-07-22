# 視線でカーソル操作＆ハンドジェスチャーでクリック操作
### （視線でカーソル操作＆まばたきでクリックの技術があるが、まばたきは生理現象なので、トリガーに使用するには現実的ではない）

## gazetrackingを採用する際の問題点
① 個人差（目の大きさ、顔の位置、画面との距離など）を考慮していないため、デフォルトでは精度が低い
-> キャリブレーション処理を自作して、視線→画面座標のマッピングを補正する
➤ 簡易キャリブレーション手法:
・画面の中央・四隅など数点を見させる
・それぞれの時のGaze値（たとえば pupil中心座標や目の角度）を記録
・それらとスクリーン座標の対応から線形回帰 or 補間関数を作成
・実行時には、そのモデルを使って変換

② 顔のアスペクト比やカメラ位置により、縦方向（上下）の変位が小さくなることがある
-> pyautogui.moveTo(x, y)の**yの値にスケーリング補正をかける**
例: y_scaled = y * 1.5など調整してバランスを取る
もしくは、回帰モデルでx, yを個別に補正 

③ 顔が上下左右に動くと、視線の相対位置が変化し、誤差が増える
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
