# Human Detection and Action Recognition with MediaPipe

MediaPipe Pose Landmarker と TensorFlow を利用して、人間の姿勢推定と行動分類を行う Python プロジェクトです。

動画やWebカメラの映像から骨格情報を抽出し、CNNによって行動を分類します。

## 主な機能

- 動画から画像データセットを生成
- MediaPipeによる骨格検出
- TensorFlow + Optunaによる行動分類モデルの学習
- 単一画像での推論
- 動画全体での推論
- Webカメラによるリアルタイム推論

---

## ディレクトリ構成

```text
Human Detection and Action Recognition with MediaPipe/
├── README.md
├── Index/
│   ├── config.py
│   ├── mediapipe_image.py
│   ├── mediapipe_video.py
│   ├── realtime_mediapipe.py
│   ├── train.py
│   ├── video2img.py
│   ├── pose_landmarker_lite.task
│   └── my_model.keras
├── Dataset/
│   ├── Jump/
│   ├── Run/
│   ├── Stand/
│   └── Walk/
├── Sample/
│   └── sample_video.mp4
├── Graph/
│   ├── optuna_results.xlsx
│   ├── precision_recall_f1.png
│   └── training_accuracy_loss.png
└── Docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── requirements.txt
```

---

## 動作環境

- Python 3.10 以上
- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- Optuna
- tqdm
- matplotlib
- japanize_matplotlib

---

## 1. データセット作成

動画から学習用画像を生成します。

```powershell
python video2img.py
```

保存先

```text
dataset/
└── Stand/
```

必要に応じて保存先フォルダ名を変更してください。

---

## 2. AIモデルの学習

Optunaを利用してハイパーパラメータ探索を行い、最適なCNNモデルを作成します。

```powershell
python train.py
```

生成されるファイル

```text
my_model.keras
optuna_results.xlsx
precision_recall_f1.png
training_accuracy_loss.png
```

---

## 3. 単一フレームの推論

動画の指定フレームを解析します。

```powershell
python mediapipe_image.py
```

出力

- 元画像
- 骨格画像
- オーバーレイ画像
- 推論結果

---

## 4. 動画全体の推論

動画の全フレームを解析します。

```powershell
python mediapipe_video.py
```

表示内容

- 進捗バー
- 各フレームの推論結果
- ラベルの出現回数

---

## 5. リアルタイム推論

Webカメラを利用してリアルタイムに姿勢推定を行います。

```powershell
python realtime_mediapipe.py
```

---

## config.py

設定を変更できます。

```python
input_video_path = "./Sample/sample_video.mp4"

labels = ["分類ラベル名1","分類ラベル名2",.....]
```

設定可能項目

- frame_number
- input_video_path
- output_image_path
- output_remove_noize_image_path
- output_overlay_image_path
- n_trials
- max_epochs
- batch_sizes
- dropout_rate
- learning_rate
- labels

---

## 処理の流れ

```text
動画
 ↓
video2img.py
 ↓
dataset作成
 ↓
train.py
 ↓
my_model.keras作成
 ↓
mediapipe_video.py
 ↓
行動分類
```

---

## 使用技術

- Python
- MediaPipe
- TensorFlow
- OpenCV
- Optuna
- NumPy

---

## 注意事項

- MediaPipeのモデルファイル（pose_landmarker_lite.task）が必要です。
- 日本語パス環境では、一時フォルダへコピーして実行する処理を実装しています。
- labelの順番は、ディレクトリの順番と同じにしてください。
