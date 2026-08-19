#!/bin/bash
# HunyuanVideo-v4 超参调参脚本
# 在多个GPU上并行跑不同的 (NUM_BLOCKS, HIDDEN_DIM) 组合
# 一个GPU跑完一个任务就自动领取下一个
#
# Usage:
#   bash run_sweep.sh --gpus 2,3,4,5,6,7
#   bash run_sweep.sh --gpus 0,1 --blocks 1,2 --hiddens 64,128,256
#   bash run_sweep.sh --gpus 4,5,6,7 --blocks 1 --hiddens 64,128

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ========== 默认参数 ==========
# 优先使用 SLURM 分配的 GPU (CUDA_VISIBLE_DEVICES)，否则默认全部8卡
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
BLOCKS_STR="1,2,3,4"
HIDDENS_STR="256"
INTERVALS_STR="6"

# GPU_IDS="0,1"
# BLOCKS_STR="1,2"
# HIDDENS_STR="256"

# ========== 解析命令行参数 ==========
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)    GPU_IDS="$2";     shift 2 ;;
        --blocks)  BLOCKS_STR="$2";  shift 2 ;;
        --hiddens) HIDDENS_STR="$2"; shift 2 ;;
        --intervals) INTERVALS_STR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_sweep.sh --gpus 0,1,2,3 [--blocks 1,2] [--hiddens 64,128,256]"
            echo ""
            echo "参数:"
            echo "  --gpus     GPU ID列表，逗号分隔 (必需)"
            echo "  --blocks   NUM_BLOCKS候选值 (默认: 1,2)"
            echo "  --hiddens  HIDDEN_DIM候选值 (默认: 64,128,256)"
            echo "  --intervals INTERVALS值 (默认: 3,4,5,6,7,8,9,10)"
            exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ========== 构建任务列表 ==========
IFS=',' read -ra BLOCKS_LIST <<< "$BLOCKS_STR"
IFS=',' read -ra HIDDENS_LIST <<< "$HIDDENS_STR"
IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"

JOBS=()
for b in "${BLOCKS_LIST[@]}"; do
    for h in "${HIDDENS_LIST[@]}"; do
        JOBS+=("${b}:${h}")
    done
done

TOTAL_JOBS=${#JOBS[@]}
NUM_GPUS=${#GPU_LIST[@]}

echo "========================================"
echo "HunyuanVideo-v4 超参调参"
echo "========================================"
echo "GPU:      ${GPU_LIST[*]} (${NUM_GPUS} 个)"
echo "INTERVALS: ${INTERVALS_STR}"
echo "任务总数: ${TOTAL_JOBS}"
echo "参数网格:"
for job in "${JOBS[@]}"; do
    b=$(echo "$job" | cut -d: -f1)
    h=$(echo "$job" | cut -d: -f2)
    echo "  NUM_BLOCKS=$b  HIDDEN_DIM=$h"
done
echo "========================================"
echo ""

# ========== 任务队列 (原子操作) ==========
JOB_INDEX_FILE=$(mktemp /tmp/sweep_idx_XXXXXX)
echo "0" > "$JOB_INDEX_FILE"
LOCK_FILE="${JOB_INDEX_FILE}.lock"

# 日志目录
LOG_DIR="${SCRIPT_DIR}/sweep_logs_8gpu_interval3-10"
mkdir -p "$LOG_DIR"

# 结果记录文件
RESULT_FILE="${LOG_DIR}/results.txt"
echo "# Sweep started at $(date)" > "$RESULT_FILE"

# ========== 清理函数 ==========
cleanup() {
    echo ""
    echo "收到中断信号，停止所有任务..."
    kill -- -$$ 2>/dev/null
    rm -f "$JOB_INDEX_FILE" "$LOCK_FILE"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ========== Worker 函数 ==========
worker() {
    local gpu_id=$1

    while true; do
        # 原子地获取下一个任务索引
        local idx
        {
            flock -x 200
            idx=$(cat "$JOB_INDEX_FILE")
            if [ "$idx" -ge "$TOTAL_JOBS" ]; then
                # 没有更多任务
                true
            else
                echo $((idx + 1)) > "$JOB_INDEX_FILE"
            fi
        } 200>"$LOCK_FILE"

        if [ "$idx" -ge "$TOTAL_JOBS" ]; then
            break
        fi

        local job="${JOBS[$idx]}"
        local num_blocks=$(echo "$job" | cut -d: -f1)
        local hidden_dim=$(echo "$job" | cut -d: -f2)
        local log_file="${LOG_DIR}/b${num_blocks}_h${hidden_dim}_interval${INTERVALS_STR}.log"

        echo "[GPU $gpu_id] 开始 ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  日志: $log_file"

        local start_time=$(date +%s)

        # 通过环境变量传参，调用 run_train.sh
        CUDA_VISIBLE_DEVICES=$gpu_id \
        NUM_BLOCKS=$num_blocks \
        HIDDEN_DIM=$hidden_dim \
        INTERVALS=$INTERVALS_STR \
            bash "${SCRIPT_DIR}/run_train.sh" \
            > "$log_file" 2>&1
        local exit_code=$?

        local end_time=$(date +%s)
        local elapsed=$(( (end_time - start_time) / 60 ))

        if [ $exit_code -eq 0 ]; then
            echo "[GPU $gpu_id] 完成 ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  (${elapsed}分钟)"
            {
                flock -x 201
                echo "OK   blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id  ${elapsed}min" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        else
            echo "[GPU $gpu_id] 失败 ($((idx+1))/$TOTAL_JOBS): blocks=$num_blocks, hidden=$hidden_dim  exit=$exit_code  查看: $log_file"
            {
                flock -x 201
                echo "FAIL blocks=$num_blocks hidden=$hidden_dim  gpu=$gpu_id  exit=$exit_code" >> "$RESULT_FILE"
            } 201>"${RESULT_FILE}.lock"
        fi
    done

    echo "[GPU $gpu_id] Worker 退出，无更多任务"
}

# ========== 启动所有 Worker ==========
for gpu in "${GPU_LIST[@]}"; do
    worker "$gpu" &
done

# 等待所有 Worker 完成
wait

# ========== 汇总结果 ==========
echo ""
echo "========================================"
echo "所有 $TOTAL_JOBS 个调参任务已完成"
echo "========================================"
echo ""
cat "$RESULT_FILE"
echo ""
echo "日志目录: $LOG_DIR"
echo "Output directory: ${SCRIPT_DIR}/outputs/"

# 清理临时文件
rm -f "$JOB_INDEX_FILE" "$LOCK_FILE" "${RESULT_FILE}.lock"
