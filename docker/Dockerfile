# ベースイメージ
FROM python:3.10-slim

# OpenCVに必要なシステムライブラリをインストール
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存ライブラリのコピーとインストール
COPY docker_env/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ソースコードのコピー
COPY . .

# 実行コマンド（例：index.pyを実行する場合）
CMD ["python", "index.py"]