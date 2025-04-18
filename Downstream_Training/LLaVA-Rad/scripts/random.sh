#!/bin/bash

CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "Number of GPUs: $CHUNKS"

for (( idx=; idx<$CHUNKS; idx++ ))
do
