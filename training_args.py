from transformers import TrainingArguments

training_arguments = TrainingArguments(
    output_dir = "./results",
    per_device_train_batch_size = 4,
    max_steps = 100
)