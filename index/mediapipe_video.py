import mediapipe as mp
import numpy as np
import shutil
import cv2
import time 
import tensorflow as tf
import config
from tqdm import tqdm
from tempfile import gettempdir
from pathlib import Path
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionRunningMode
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarker, FaceLandmarkerOptions
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

MODEL_PATH = Path(__file__).resolve().parent / "pose_landmarker_lite.task"
FACE_MODEL_PATH = Path(__file__).resolve().parent / "face_landmarker.task"
MODEL_CACHE_DIR = Path(gettempdir()) / "mediapipe_models"
DATASET_DIR = Path(__file__).resolve().parent / ".." / "dataset"
MODEL_CANDIDATES = (
    Path(__file__).resolve().parent / "model.keras",
    Path(__file__).resolve().parent / "my_model.keras",
    Path.cwd() / "model.keras",
    Path.cwd() / "my_model.keras",
)

def video_capture(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total_frames

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

def run_mediapipe():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {MODEL_PATH}")
    runtime_model_path = prepare_runtime_model_path(MODEL_PATH)
    if runtime_model_path is None:
        raise RuntimeError("モデルファイルの準備に失敗しました")
    pose_options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(runtime_model_path)),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_landmarker = PoseLandmarker.create_from_options(pose_options)

    face_landmarker = None
    if FACE_MODEL_PATH.exists():
        runtime_face_model_path = prepare_runtime_model_path(FACE_MODEL_PATH)
        if runtime_face_model_path is None:
            raise RuntimeError("モデルファイルの準備に失敗しました")
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(runtime_face_model_path)),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=0,
            min_face_detection_confidence=0,
            min_tracking_confidence=0,
        )
        face_landmarker = FaceLandmarker.create_from_options(face_options)
    else:
        print(f"face model not found, skipping face landmarker: {FACE_MODEL_PATH}")

    return pose_landmarker, face_landmarker

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


def resolve_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            return candidate
    searched = "\n".join(str(path) for path in MODEL_CANDIDATES)
    raise FileNotFoundError(f"モデルファイルが見つかりません。次を確認してください:\n{searched}")


def resolve_labels() -> list[str]:
    if DATASET_DIR.exists():
        class_dirs = sorted([path.name for path in DATASET_DIR.iterdir() if path.is_dir()])
        if class_dirs:
            return class_dirs
    return list(config.labels)

def CNN_cluster_image(model, input_image, labels):
    resized_img = cv2.resize(input_image, (128, 128))
    normalized_img = resized_img.astype(np.float32) / 255.0
    batched_img = np.expand_dims(normalized_img, axis=0)
    predictions = model.predict(batched_img, verbose=0)
    class_index = np.argmax(predictions)
    return labels[class_index]
    
def main():
    pose_landmarker, face_landmarker = run_mediapipe()
    # cnn_model = CNN()
    cnn_model = tf.keras.models.load_model(resolve_model_path())
    cap, fps, total_frames = video_capture(config.input_video_path)
    frame_count = 0
    bar = tqdm(total = total_frames)

    labels = resolve_labels()
    label_count = {label: 0 for label in labels}
    print(f"使用ラベル: {labels}")
    
    
    print(f"総フレーム数：{total_frames}")
    time.sleep(1)
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        bar.n = frame_count
        bar.update()
        
        if frame_count <= total_frames:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = image_rgb)
            timestamp_ms = int((frame_count / fps) * 1000)
            
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            
            if pose_result.pose_landmarks:
                if face_landmarker is not None:
                    face_landmarker.detect_for_video(mp_image, timestamp_ms)                                            
                remove_noize_image = remove_noize_from_image(frame, pose_result)
                overlay_image = cv2.addWeighted(frame, 0.7, remove_noize_image, 1.0, 0)
                cv2.imwrite(config.output_image_path, image_rgb)
                cv2.imwrite(config.output_remove_noize_image_path, remove_noize_image)
                cv2.imwrite(config.output_overlay_image_path, overlay_image)
                cluster_cnn = CNN_cluster_image(cnn_model, image_rgb, labels)
                label_count[cluster_cnn] += 1
                bar.write(f"フレーム: {frame_count} | 検出: {cluster_cnn}")
                                
            else:
                pass
        else:
            break
            
    cap.release()
    pose_landmarker.close()
    if face_landmarker is not None:
        face_landmarker.close()
    bar.close()
    
    print(f"総フレーム数：{total_frames}")
    print("各ラベルの出現回数")
    for label , count in label_count.items():
        print(f"{label}:{count}")

main()