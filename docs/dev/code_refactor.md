# Argument Parsing Refactor 


## Backtround 

The current argument parsing system is a bit convoluted and may be complex to maintain.

## Environment 
We use conda env `llmhalluc` to run the codebase.

## High Level Design 

- We should support load from cli and yaml file together. The priority is cli > yaml > default defined in the argument class. 
- We should keep the yaml file sample, that means the yaml file should only contains the arguments that are different from the default values. The current `e2e.yaml` is a good example. 
- When we need one additional argument, we should only add it once. Currently, when we need an additional argument, we should pass it to TrainArgumetns and the specific FTArguments under `ft_args.py`, which is nto a good practice.



## Structure Output

The output_dir should be follow the current logic. 
Specifically, without specific defined, the output_dir should be:
`./outputs/{model_name}/{run_name}/{finetuning_type}/{mode}`
where mode is `train` or `eval`. 

Mention that for eval mode, we should add pass `./outputs/{model_name}/{run_name}/{finetuning_type}/eval/{tasks}/results.json` as the output_dir.

Here, the model_name should be the follow the current logic in lower case, i.e., `meta-llama/Llama-3.2-1b` => `llama-3.2-1b`, if not specified.
The run_name is a required argument and should not be specified in the dataclass. Afterward, we post process the run_name with `{model_name}_{run_name}_{stage}`

The config_path should be follow the current logic, i.e., `./outputs/{model_name}/{run_name}/{finetuning_type}/{mode}_config.yaml`

We should also include the eval script like `eval.sh` save in the output directory, i.e., `./outputs/{model_name}/{run_name}/{finetuning_type}/eval.sh`

You can review the current logic in the codebase to understand the logic, and take current `outputs` folder to understand the output structure.

We should not save the special_token_config.json in the output directory anymore. 


## Features we should support

We should support Peft configs, early stopping callback, and special token initialization and replacement. 
Ultimately, we should not change any code except for argument loading, parsing and coding. 


## Instructions

You should act as an expert in machine learning engineering and familiar with training LLMs with huggingface packages, also use lm_eval for evaluation.

You should also be familiar with the codebase and the current argument parsing system.
