from transformers import AutoModelForCausalLM, AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


print(len(tokenizer))
print(model.get_input_embeddings().weight.size(0))
print(model.get_output_embeddings().weight.size(0))
print(model.config.vocab_size)


tokenizer.add_special_tokens({"additional_special_tokens": ["<|BACKTRACK|>"]})
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)


print(len(tokenizer))
print(model.get_input_embeddings().weight.size(0))
print(model.get_output_embeddings().weight.size(0))
print(model.config.vocab_size)


print(tokenizer.convert_tokens_to_ids("<|BACKTRACK|>"))
print(model.config.tie_word_embeddings)
