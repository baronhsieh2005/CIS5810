# train_vit.py
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate

# 1. Load dataset
dataset = load_dataset("imagefolder", data_dir="./custom_dataset", split="train")
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
val_dataset = dataset["test"]
# 2. Load pretrained ViT
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,  # Correct / Incorrect
    id2label={0: "incorrect", 1: "correct"},
    ignore_mismatched_sizes=True
)

# 3. Preprocess images
processor = ViTImageProcessor.from_pretrained(model_name)


def transform(example_batch):
    # Process a batch of images
    inputs = processor(example_batch["image"], return_tensors="pt")
    inputs["labels"] = example_batch["label"]
    return inputs


dataset = dataset.with_transform(transform)

# 4. Metrics
accuracy = evaluate.load("accuracy")


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return accuracy.compute(predictions=preds, references=p.label_ids)


# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    push_to_hub=False,
    remove_unused_columns=False
)

# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    compute_metrics=compute_metrics
)

# 7. Train
trainer.train()

# 8. Save model
trainer.save_model("./vit_squat_model")
