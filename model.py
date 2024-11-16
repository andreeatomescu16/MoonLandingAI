import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

llama_model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path = "aboonaji/llama2finetune-v2"
).to(device)



llama_model.config.use_cache = False
llama_model.config.pretraining_tp = 1