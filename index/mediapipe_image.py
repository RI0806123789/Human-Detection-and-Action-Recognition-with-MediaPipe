import mediapipe as mp
import numpy as np
import shutil
import cv2
import tensorflow as tf
import config
from tempfile import gettempdir
from pathlib import Path
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"
MODEL_CACHE_DIR = Path(gettempdir()) / "mediapipe_models"

def image_from_video(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            raise ValueError
    finally:
        cap.release()
    return frame # BGRの順番で出力する

def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def prepare_runtime_model_path(model_path: Path) -> Path | None:
    if is_ascii_path(model_path):
        return model_path
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runtime_model_path = MODEL_CACHE_DIR / model_path.name
    try:
        if (not runtime_model_path.exists()) or (runtime_model_path.stat().st_size != model_path.stat().st_size):
            shutil.copyfile(model_path, runtime_model_path)
    except OSError as error:
        print(f"モデルのコピーに失敗しました: {error}")
        return None
    return runtime_model_path

def run_mediapipe(image):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")
    runtime_model_path = prepare_runtime_model_path(MODEL_PATH)
    if runtime_model_path is None:
        raise RuntimeError("モデルファイルの準備に失敗しました")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(runtime_model_path)),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.5,
    )
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    with PoseLandmarker.create_from_options(options) as landmarker:
        return landmarker.detect(mp_image)


def remove_noize_from_image(image, pose_result):
    background = np.zeros(image.shape, dtype = np.uint8)
    h, w , _ = image.shape
    POSE_CONNECTIONS = frozenset([
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
        (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
    ])
    for pose_landmarks in pose_result.pose_landmarks:
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_point = pose_landmarks[start_idx]
            end_point = pose_landmarks[end_idx]

            sx, sy = int(start_point.x * w), int(start_point.y * h)
            ex, ey = int(end_point.x * w), int(end_point.y * h)
            cv2.line(background, (sx, sy), (ex, ey), (245, 66, 230), 2)
        for landmark in pose_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(background, (cx, cy), 3, (245, 117, 66), -1)
    no_noize_image = cv2.medianBlur(background, 3)
    return no_noize_image    
    
def CNN():
    model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(128, 128, 3)),
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation= "relu"),
            tf.keras.layers.Dense(4, activation= "softmax")
    ])
    return model

def CNN_cluster_image(model, remove_noize_image, labels):
    resized_img = cv2.resize(remove_noize_image, (128, 128))
    normalized_img = resized_img / 255.0
    batched_img = np.expand_dims(normalized_img, axis=0)
    predictions = model.predict(batched_img, verbose=0)
    class_index = np.argmax(predictions)
    return labels[class_index]

def output_result_text(cluster_cnn):
    print(f"\n検出結果：{cluster_cnn}\n")
  
def main():
    img = image_from_video(config.input_video_path, config.frame_number)
    pose_results = run_mediapipe(img)
    print(f"detected poses: {len(pose_results.pose_landmarks)}")
    print(pose_results)
    remove_noize_image  = remove_noize_from_image(img, pose_results)
    print(remove_noize_image)
    overlay_image = cv2.addWeighted(img, 0.7, remove_noize_image, 1.0, 0)
    cv2.imwrite(config.output_image_path, img)
    cv2.imwrite(config.output_remove_noize_image_path, remove_noize_image)
    cv2.imwrite(config.output_overlay_image_path, overlay_image)
    model = CNN()
    cluster_cnn = CNN_cluster_image(model, remove_noize_image, config.labels)
    output_result_text(cluster_cnn)

main()

"""
開発フロー
frame_number = 3枚でも動くか？？
↓
動画版に移行する
↓
labelsの内容を本格的なものに変更する
↓
webotsコードと融合する
↓
VLAライクに動作するように変更する
"""
