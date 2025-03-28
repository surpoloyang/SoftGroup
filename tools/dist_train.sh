#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-29500}

OMP_NUM_THREADS=1 torchrun --nproc_per_node=$GPUS --master_port=$PORT $(dirname "$0")/train.py --dist $CONFIG ${@:3}   # $CONFIG <- $1，即传入的第一个参数传入到train.py中作为args.config; ${@:3}是将第3个参数到最后一个参数传入到train.py中比如--resume, --work_dir, --skip_validate等
