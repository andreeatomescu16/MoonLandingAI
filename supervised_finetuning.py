from trl import SFTTrainer
from model import *
from training_args import *
from peft import LoraConfig
from datasets import load_dataset
from tokenizer_loader import *

llama_sft_trainer = SFTTrainer(
    model = llama_model,
    args = training_arguments,
    train_dataset = load_dataset(
        path = "aboonaji/wiki_medical_terms_llam2_format",
        split = "train"
    ),
    tokenizer = llama_tokenizer,
    peft_config = LoraConfig(task_type = "CASUAL_LM", r = 64, lora_alpha = 16, lora_dropout = 0.1),
    dataset_text_field = "text"
)