#!/bin/bash

LOG_DIR="./logs"
find "$LOG_DIR" -type f -name "*.log" -mmin +30 -exec rm -f {} \;

