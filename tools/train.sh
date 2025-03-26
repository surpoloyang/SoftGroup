#!/usr/bin/env bash
# CONFIG=$1
# OMP_NUM_THREADS=1 python $(dirname "$0")/train.py $CONFIG ${@:2}   

OMP_NUM_THREADS=1 python $(dirname "$0")/train.py   