#!/bin/bash

conda create -n lm_eval python=3.11 -y
conda activate lm_eval
pip install -e ./lm-evaluation-harness/
