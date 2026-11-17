#!/bin/bash
cd /root/models/bge-m3
echo "=== 尝试下载模型权重 ==="
wget -c https://hf-mirror.com/BAAI/bge-m3/resolve/main/pytorch_model.bin
echo "=== 完成 ==="
ls -lh /root/models/bge-m3/
