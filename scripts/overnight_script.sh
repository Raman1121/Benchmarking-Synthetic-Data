#!/bin/bash

source /raid/s2198939/miniconda3/bin/activate sana

echo "Calcuating conditional metrics for RadEdit"

./scripts/image_quality_metrics_conditional.sh

echo "Calculating regular metrics for Sana ckpt13 and Pixart ckpt20"
./scripts/image_quality_metrics_memory_saving.sh