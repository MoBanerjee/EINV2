#!/bin/bash

GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python3 code/main.py train --port=12360