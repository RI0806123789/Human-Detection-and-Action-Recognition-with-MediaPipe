import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示用
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import pandas as pd
import numpy as np
import config

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# 【固定パラメータ】
IMG_HEIGHT = config.height
IMG_WIDTH = config.width
MAX_EPOCHS = config.max_epochs   

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
    # Deprecated: original function made only a 2-way split and reused the same
    # "test" set as validation during Optuna tuning. Replace usage with
    # `split_dataset` + `create_tf_dataset` for explicit train/val/test splits.
    indices = np.arange(len(file_paths))
    rng = np.random.default_rng(123)
    rng.shuffle(indices)

    shuffled_file_paths = [file_paths[index] for index in indices]
    shuffled_labels = [labels[index] for index in indices]

    val_count = int(len(file_paths) * 0.2)
    test_file_paths = shuffled_file_paths[:val_count]
    test_labels = np.array(shuffled_labels[:val_count])
    train_file_paths = shuffled_file_paths[val_count:]
    train_labels = np.array(shuffled_labels[val_count:])

    train_data = tf.data.Dataset.from_tensor_slices((train_file_paths, train_labels))
    train_data = train_data.map(decode_and_resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_data = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
    test_data = test_data.map(decode_and_resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_data, test_data, test_labels


def split_dataset(file_paths, labels, seed=123, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Stratified split into train/val/test by class (returns lists).

    Keeps class proportions by splitting per-class. Returns
    (train_paths, train_labels, val_paths, val_labels, test_paths, test_labels).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    file_paths = np.array(file_paths)
    labels = np.array(labels)
    rng = np.random.default_rng(seed)

    train_idx = []
    val_idx = []
    test_idx = []

    classes = np.unique(labels)
    for c in classes:
        idxs = np.where(labels == c)[0].tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test
        if n_train < 0:
            # fallback when very small class counts: allocate at least one where possible
            n_train = max(0, n - (n_val + n_test))
        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train:n_train + n_val])
        test_idx.extend(idxs[n_train + n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    train_paths = file_paths[train_idx].tolist()
    train_labels = labels[train_idx].tolist()
    val_paths = file_paths[val_idx].tolist()
    val_labels = labels[val_idx].tolist()
    test_paths = file_paths[test_idx].tolist()
    test_labels = labels[test_idx].tolist()

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def create_tf_dataset(file_paths, labels, batch_size):
    labels = np.array(labels)
    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds = ds.map(decode_and_resize).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def calculate_classification_metrics(y_true, y_pred, class_names):
    confusion = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(class_names)).numpy()
    metrics = []

    for class_index, class_name in enumerate(class_names):
        true_positive = confusion[class_index, class_index]
        false_positive = confusion[:, class_index].sum() - true_positive
        false_negative = confusion[class_index, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
        f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics.append({
            "class_name": class_name,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        })

    return metrics

def plot_classification_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = [item["class_name"] for item in metrics]
    precision = [item["precision"] for item in metrics]
    recall = [item["recall"] for item in metrics]
    f1_score = [item["f1_score"] for item in metrics]

    x = np.arange(len(class_names))
    width = 0.25

    plt.figure(figsize=(max(10, len(class_names) * 1.5), 6))
    plt.bar(x - width, precision, width, label="適合率 (Precision)")
    plt.bar(x, recall, width, label="再現率 (Recall)")
    plt.bar(x + width, f1_score, width, label="F値 (F1-score)")
    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.ylabel("スコア")
    plt.title("クラス別 Precision / Recall / F1-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_training_history(history, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss')
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='訓練')
    if val_acc is not None:
        plt.plot(epochs_range, val_acc, label='検証')
    plt.title('正解率の推移')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='訓練')
    if val_loss is not None:
        plt.plot(epochs_range, val_loss, label='検証')
    plt.title('損失の推移')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def objective(trial, train_paths, train_labels, val_paths, val_labels, num_classes):
    batch_size = trial.suggest_categorical('batch_size', config.batch_sizes)
    hidden_units = trial.suggest_int('hidden_units', config.hidden_units[0], config.hidden_units[1], step=config.hidden_units[2])
    dropout_rate = trial.suggest_float('dropout_rate', config.dropout_rate[0], config.dropout_rate[1])
    learning_rate = trial.suggest_float('learning_rate', config.learning_rate[0], config.learning_rate[1], log=True)

    # create tf.data datasets for this trial (only batch size depends on trial)
    train_data = create_tf_dataset(train_paths, train_labels, batch_size)
    val_data = create_tf_dataset(val_paths, val_labels, batch_size)

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
        validation_data=val_data,
        epochs=MAX_EPOCHS,
        callbacks=[early_stopping],
        verbose=1  # 0にすると、裏で静かに学習してくれます
    )

    tf.keras.backend.clear_session()

    best_val_acc = max(history.history['val_accuracy'])
    return best_val_acc

def main():
    # 画像の読み込み
    base_dir = Path(__file__).resolve().parent / ".."
    dataset_dir = base_dir / "Dataset"
    file_paths, labels, class_names = collect_image_paths(dataset_dir)
    num_classes = len(class_names)

    # create a single deterministic stratified split: train/val/test = 60/20/20
    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = (
        split_dataset(file_paths, labels, seed=123, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    )

    print("Optunaによるパラメータ自動探索を開始します。")
    study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, train_paths, train_labels, val_paths, val_labels, num_classes), n_trials=config.n_trials)

    df_results = study.trials_dataframe()
    output_dir = base_dir / "Graph"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_results.to_excel(output_dir / "optuna_results.xlsx", index=False)
    print("\n全実験データは 'optuna_results.xlsx' に保存されました。")

    print("\n見つけたパラメータで最終モデルを学習します")
    best_params = study.best_params
    print("パラメータ:", best_params)

    # 最強パラメータを取り出す
    best_batch = best_params['batch_size']
    best_hidden = best_params['hidden_units']
    best_dropout = best_params['dropout_rate']
    best_lr = best_params['learning_rate']

    # 最終モデルの構築 -- 再学習は train+val で行い、test は最後に一度だけ使う
    train_plus_paths = train_paths + val_paths
    train_plus_labels = train_labels + val_labels
    train_data = create_tf_dataset(train_plus_paths, train_plus_labels, best_batch)
    test_data = create_tf_dataset(test_paths, test_labels, best_batch)

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

    # Re-train final model on train+val (do not peek at test during training).
    history = final_model.fit(
        train_data,
        epochs=MAX_EPOCHS,
        verbose=1
    )

    model_dir = Path(__file__).resolve().parent
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "my_model.keras"
    final_model.save(model_path)
    print(f" AIモデルを '{model_path.name}' として保存しました")

    test_predictions = final_model.predict(test_data, verbose=0)
    predicted_labels = np.argmax(test_predictions, axis=1)
    metrics = calculate_classification_metrics(test_labels, predicted_labels, class_names)
    metrics_df = pd.DataFrame(metrics)
    print("\nクラス別評価指標")
    print(metrics_df.round(4).to_string(index=False))

    summary_metrics = metrics_df[["precision", "recall", "f1_score"]].mean().round(4)
    print("\n平均評価指標")
    print(summary_metrics.to_string())

    plot_classification_metrics(metrics, output_dir / "precision_recall_f1.png")
    print("評価グラフを 'precision_recall_f1.png' として保存しました")

    plot_training_history(history, output_dir / "training_accuracy_loss.png")
    print("学習曲線を 'training_accuracy_loss.png' として保存しました")

if __name__ == "__main__":
    main()
