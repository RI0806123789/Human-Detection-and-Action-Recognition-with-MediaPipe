import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示用
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pandas as pd

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# 【固定パラメータ】
IMG_HEIGHT = 128 # 画像の高さ
IMG_WIDTH = 128  # 画像の幅
MAX_EPOCHS = 25  # 最大学習エポック数

def collect_image_paths(dataset_dir):
    dataset_dir = Path(dataset_dir)
    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    file_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        for path in class_dir.rglob("*"):
            if path.suffix.lower() in ALLOWED_EXTS:
                file_paths.append(str(path))
                labels.append(label)
    return file_paths, labels, class_names

def decode_and_resize(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def create_dataset(file_paths, labels, batch_size):
    data = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    data = data.shuffle(len(file_paths), seed=123, reshuffle_each_iteration=False)

    val_count = int(len(file_paths) * 0.2)
    train_data = data.skip(val_count).map(decode_and_resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_data = data.take(val_count).map(decode_and_resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_data, test_data

def objective(trial, file_paths, labels, num_classes):
    # ① 探してほしいパラメータの範囲を指定
    batch_size = trial.suggest_categorical('batch_size', [4,8,16,32, 64, 128,256,512,1024])
    hidden_units = trial.suggest_int('hidden_units', 32, 512, step=32)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
    learning_rate = trial.suggest_float('learning_rate', 1e-10, 1e-2, log=True)

    train_data, test_data = create_dataset(file_paths, labels, batch_size)

    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(num_classes, activation="softmax") 
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=0  # 0にすると、裏で静かに学習してくれます
    )

    tf.keras.backend.clear_session()

    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc

def main():
    # 画像の読み込み
    dataset_dir = Path(__file__).resolve().parent / ".." / "dataset"
    file_paths, labels, class_names = collect_image_paths(dataset_dir)
    num_classes = len(class_names)

    print("--- Optunaによるパラメータ自動探索を開始します ---")
    study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, file_paths, labels, num_classes), n_trials=10)

    df_results = study.trials_dataframe()
    df_results.to_excel("optuna_results.xlsx", index=False)
    print("\n最適化完了！全実験データは 'optuna_results.xlsx' に保存されました。")

    print("\n見つけた最強のパラメータで最終モデルを学習します...")
    best_params = study.best_params
    print("最強パラメータ:", best_params)

    # 最強パラメータを取り出す
    best_batch = best_params['batch_size']
    best_hidden = best_params['hidden_units']
    best_dropout = best_params['dropout_rate']
    best_lr = best_params['learning_rate']

    train_data, test_data = create_dataset(file_paths, labels, best_batch)

    # 最終モデルの構築
    final_model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomZoom(0.05),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(best_hidden, activation="relu"),
        tf.keras.layers.Dropout(best_dropout),
        tf.keras.layers.Dense(num_classes, activation="softmax") 
    ])

    final_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_lr),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = final_model.fit(
        train_data,
        validation_data=test_data,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    final_model.save("my_best_model.keras")
    print(" 最強のAIモデルを 'my_best_model.keras' として保存しました！")

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='訓練')
    plt.plot(epochs_range, val_acc, label='検証')
    plt.title('正解率の推移')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='訓練')
    plt.plot(epochs_range, val_loss, label='検証')
    plt.title('損失の推移')
    plt.legend()
    plt.show()

main()