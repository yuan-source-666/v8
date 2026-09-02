#!/usr/bin/env bash
# v8 一键启动脚本（Linux/macOS）：先 prepare 后 train
# 用法：bash scripts/run_mini.sh        # 也支持 V8_CONFIG/V8_PROFILE 环境变量
set -e
cd "$(dirname "$0")/.."

CONFIG="${V8_CONFIG:-config/mini.yaml}"
PROFILE="${V8_PROFILE:-mini}"

echo "[run_mini] 第 1 步：数据管线（prepare）"
python data/prepare.py --config "$CONFIG" --profile "$PROFILE" --demo

echo "[run_mini] 第 2 步：预训练（train，启用 drives）"
python train/train.py --config "$CONFIG" --profile "$PROFILE" --use-drives
