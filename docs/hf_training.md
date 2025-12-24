# Train Pipeline using HuggingFace


## Background
In this project, we used to use llamafactory for training. However, we find that there are some bugs in training with llamafactory. So, we switch to use transformers trainers or trl SFTTrainer to train directly. However, we want to maintain the useability of train with llamafactory. 
When training with huggingface trainer, the entry point is @llmhalluc/hf.py, and we train to maximize the compatiability to use the current pipeline (current pipeline, i.e. load args, etc are for llamafactory). So we introduce an new function hf_cfg_setup, and several run workflow to support the training with huggingface trainer.

Besides that, we also has several scripts and data loading logic to prepare the dataset for llamafactory training or huggingface trainer. 


## Your tasks.

You task is to make a implementation plan for the following tasks:
- [] Review the current codebase, and identify any potential issues, such as recursive loading, etc. 
- [] Finish the implementation of training with SFT. The most undone part is the data loading logic under the SFTExecutor. You should try to use existing code as much as possible. 

The implementation plan should be a detailed plan, including code structures, data loading logic, and any other relevant details. Make it a markdown file and name it as `hf_training_plan.md` under the folder `docs`.

After you finished the implementation plan, you should wait for my approval before you start the implementation.