# AI Agents and Models Documentation

## Overview

This project focuses on **new autoregression generation** in large language models through innovative backtracking mechanisms. The system employs models to train, evaluate, and improve LLM performance on mathematical reasoning and question-answering tasks.

## Train Pipeline 

### Train with Transformers
Alternatively, we directly use Hugging Face Transformers for training.


## Evaluation Pipeline 

### lm-evaluation-harness for Evaluating LLMs
We use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for evaluating LLMs.

### Datasets 
We currently use GSM8K (openai/gsm8k) from huggingface for evaluation. Considering the **Backtrack** mechanism, we need to create a new task (in lm-eval), and loaded for lm-eval.


## Key Principle 

### Separate Environment for Training and Evaluation 

Due to the incompatibility of the environment from **Llama-Factory** and **lm-evaluation-harness**, we need to separate the environment for training and evaluation. We use conda to manage the environment. The environment name is `llamafactory` for training and `lm_eval` for evaluation.

### Use Python Scripts for Training and Evaluation instead of CLI.

Although Llama-Factory and lm-evaluation-harness provide CLI tools, we find it more convenient to use Python scripts for training and evaluation. This is because we need to modify the code for our own needs, and the CLI tools are not flexible enough.

### Workflow 

When you edit the code, you should update the recent changes section to reflect the recent change (keep the order of the changes and should only keep five recent changes). Moreover, if all the tasks are completed, you should change the current tasks section to 'N/A'.


## Environment 

### Conda Environment

We use conda to manage the environment, specifically, we use `llmhalluc` in conda environment.

If you find the conda is not activated, you can use the following command:
```bash
source ~/.activate_conda
```
to activate the conda environment.

