import time
import shutil
from pathlib import Path
from typing import Optional
from tempfile import gettempdir
from urllib.error import URLError
from urllib.request import Request, urlopen

import cv2
import mediapipe as mp

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_lib
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as VisionRunningMode
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions

# MediaPipe 0.10.33のPythonパッケージ側のバグを修正するためのモンキーパッチ
if not hasattr(running_mode_lib, "validate_running_mode"):
    def _validate_running_mode(running_mode, result_callback):
        pass

    running_mode_lib.validate_running_mode = _validate_running_mode


MODEL_FILE = "pose_landmarker_lite.task"
MODEL_URLS = [
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
]
WINDOW_NAME = "MediaPipe Pose Test (Windows)"
CAMERA_INDEX = 0
MODEL_CACHE_DIR = Path(gettempdir()) / "mediapipe_models"

# 描画用の骨格の接続情報
POSE_CONNECTIONS = frozenset([
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
])


def create_options() -> Optional[PoseLandmarkerOptions]:
    model_path = find_model_path()
    if model_path is None:
        return None

    runtime_model_path = prepare_runtime_model_path(model_path)
    if runtime_model_path is None:
        return None

    return PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(runtime_model_path)),
        running_mode=VisionRunningMode.VIDEO,
    )


def is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def prepare_runtime_model_path(model_path: Path) -> Optional[Path]:
    if is_ascii_path(model_path):
        return model_path

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    runtime_model_path = MODEL_CACHE_DIR / MODEL_FILE

    try:
        if (not runtime_model_path.exists()) or (runtime_model_path.stat().st_size != model_path.stat().st_size):
            shutil.copyfile(model_path, runtime_model_path)
            print(f"日本語パス回避のためモデルをコピーしました: {runtime_model_path}")
    except OSError as error:
        print(f"モデルのコピーに失敗しました: {error}")
        return None

    return runtime_model_path


def find_model_path() -> Optional[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / MODEL_FILE,
        Path.cwd() / MODEL_FILE,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    save_path = script_dir / MODEL_FILE
    if download_model(save_path):
        return save_path

    return None


def download_model(save_path: Path) -> bool:
    print(f"モデルファイルが見つからないため、自動ダウンロードを試行します: {save_path}")
    temp_path = save_path.with_suffix(save_path.suffix + ".tmp")

    for url in MODEL_URLS:
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                data = response.read()

            with open(temp_path, "wb") as temp_file:
                temp_file.write(data)

            temp_path.replace(save_path)
            print(f"モデルをダウンロードしました: {url}")
            return True
        except (URLError, OSError, TimeoutError) as error:
            print(f"ダウンロード失敗 ({url}): {error}")
            if temp_path.exists():
                temp_path.unlink()

    print("モデルの自動ダウンロードに失敗しました。手動で配置してください。")
    return False


def open_camera(index: int) -> cv2.VideoCapture:
    # WindowsではDirectShowの方が起動安定性が高い場合がある
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        return cap

    cap.release()
    # DirectShowで開けない環境向けに通常バックエンドへフォールバック
    return cv2.VideoCapture(index)


def draw_pose(image, pose_landmarks_list) -> None:
    if not pose_landmarks_list:
        return

    h, w, _ = image.shape

    for pose_landmarks in pose_landmarks_list:
        for start_idx, end_idx in POSE_CONNECTIONS:
            start_point = pose_landmarks[start_idx]
            end_point = pose_landmarks[end_idx]

            sx, sy = int(start_point.x * w), int(start_point.y * h)
            ex, ey = int(end_point.x * w), int(end_point.y * h)
            cv2.line(image, (sx, sy), (ex, ey), (245, 66, 230), 2)

        for landmark in pose_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 3, (245, 117, 66), -1)


def run_tasks_mode(cap: cv2.VideoCapture, options: PoseLandmarkerOptions) -> None:
    # VIDEOモードは単調増加タイムスタンプが必要なので、実時間ベースで生成する
    start_time = time.perf_counter()
    last_timestamp_ms = -1

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("カメラの映像が読み込めません。")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((time.perf_counter() - start_time) * 1000.0)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            pose_result = landmarker.detect_for_video(mp_image, timestamp_ms)
            draw_pose(image, pose_result.pose_landmarks)

            cv2.imshow(WINDOW_NAME, image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def main() -> None:
    options = create_options()
    cap = open_camera(CAMERA_INDEX)

    if not cap.isOpened():
        print("カメラを起動できませんでした。CAMERA_INDEXを変更して再実行してください。")
        return

    print("カメラを起動しています... ('q'キーで終了します)")
    if options is None:
        print("モデルを準備できなかったため終了します。")
        print(f"探索先: {Path(__file__).resolve().parent} と {Path.cwd()}")
        print(f"配置先ファイル名: {MODEL_FILE}")
        return

    try:
        run_tasks_mode(cap, options)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()