from transformers import pipeline
from train_model import *

user_prompt = ""
text_generation_pipeline = pipeline(
    task = "text-generation",
    model = llama_model,
    tokenizer = llama_tokenizer,
    max_length = 300
)

model_answear = text_generation_pipeline(f"<s>[INST] {user_prompt} [/INST]")
print(model_answear[0]['generated_text'])