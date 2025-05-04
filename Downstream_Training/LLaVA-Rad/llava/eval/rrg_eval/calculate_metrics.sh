#!/bin/bash

T2I_MODEL="sana"
DATA_PERCENTAGE=100
RUN_NAME="eval_${T2I_MODEL}_${DATA_PERCENTAGE}"
OUTPUT_DIR="results"

mkdir -p $OUTPUT_DIR

python run.py --t2i_model $T2I_MODEL --data_percentage $DATA_PERCENTAGE --run_name $RUN_NAME --output_dir $OUTPUT_DIR