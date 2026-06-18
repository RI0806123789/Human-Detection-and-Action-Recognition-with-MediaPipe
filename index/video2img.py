import cv2
import os
import numpy as np  
import config

current_dir = os.path.dirname(os.path.abspath(__file__))

save_dir = os.path.join(current_dir, '..', 'dataset', 'Stand')
save_dir = os.path.normpath(save_dir)

print(f"【確認1】画像の保存先パス: {save_dir}")
os.makedirs(save_dir, exist_ok=True)

print(f"【確認2】読み込む動画のパス: {config.input_video_path}")
cap = cv2.VideoCapture(config.input_video_path)

if not cap.isOpened():
    print("エラー: 動画ファイルを開けませんでした。")
    exit()

count = 0
success_count = 0  #

while True:
    ret, frame = cap.read()
    if not ret:
        print("動画の読み込みが完了し、処理を終了しました。")
        break
    
    save_path = os.path.join(save_dir, f'frame_{count:04d}.jpg')
    
    # 💡【日本語パス対策】cv2.imwriteを使わず、バイナリデータとして安全に書き込む
    try:
        # 画像をJPG形式にエンコード（メモリ上）
        ret_code, img_encode = cv2.imencode('.jpg', frame)
        if ret_code:
            # 日本語パス対応のファイル書き込み
            with open(save_path, 'wb') as f:
                f.write(img_encode)
            success_count += 1
    except Exception as e:
        if count == 0:  # 最初のエラーだけ表示
            print(f"❌ 書き込みエラーが発生しました: {e}")
            
    count += 1

print(f"動画の総フレーム数: {count} 枚")
print(f"実際にStandフォルダに保存された画像数: {success_count} 枚！")
cap.release()