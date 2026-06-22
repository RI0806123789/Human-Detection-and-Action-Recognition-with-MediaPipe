#mediapipe_image.py and mediapipe_video.py
frame_number = 50

input_video_path = "./Sample/sample_video.mp4"

output_image_path = "./Sample/sample.png"

output_remove_noize_image_path = "./Sample/sample_image_remove_noize.jpeg"

output_overlay_image_path = "./Sample/sample_image_overlay.jpeg"


#train.py
n_trials=15

batch_sizes = [2,4,8,16,32,64,128,256,512,1024]

hidden_units = 32, 1024, 32 # (min, max, step)

dropout_rate = 0.1, 0.9 # (min, max, step)

learning_rate = 1e-10, 1e-1 # (min, max, step)


# Keep label order consistent with dataset folder names used for training.
labels = ["Jump", "Run", "Stand", "Walk"]
