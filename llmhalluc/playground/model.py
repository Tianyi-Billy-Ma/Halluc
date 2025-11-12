from transformers import AutoModelForCausalLM, AutoTokenizer


model_name_or_path = "./models/qwen3-4b-instruct-2507_gsm8k_bt_sft_lora"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print(len(tokenizer))
current_embedding_size = model.get_input_embeddings().weight.size(0)
print(current_embedding_size)


text = "What does <|BACKTRACK|> means?"
