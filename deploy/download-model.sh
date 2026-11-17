#!/bin/bash
mkdir -p /root/models/bge-m3
cd /root/models/bge-m3
echo "=== 下载小文件 ==="
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/config.json
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/tokenizer.json
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/tokenizer_config.json
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/sentencepiece.bpe.model
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/special_tokens_map.json
echo "=== 下载大文件 model.safetensors (约2GB) ==="
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/model.safetensors
echo "=== 完成 ==="
ls -lh /root/models/bge-m3/
