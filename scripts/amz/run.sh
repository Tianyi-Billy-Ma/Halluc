#!/bin/bash

accelerate launch -m llmhalluc.run_train
accelerate launch -m llmhalluc.run_eval