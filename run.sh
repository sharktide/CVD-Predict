#!/bin/bash
cd /Users/rihaan.meher/Documents/CVD-Predict
python3 train_realistic.py > pipeline_realistic.log 2>&1
echo "DONE: exit code $?" >> pipeline_realistic.log
