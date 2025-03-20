import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Part 1: Fine-Tuning BERT
print("=== Part 1: Fine-Tuning BERT ===")

# Step 1: Load the IMDb dataset
dataset = load_dataset("imdb")
print("Dataset loaded:", dataset)

# Step 2: Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)

# Step 3: Preprocess the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Split into train and validation sets (80-20 split)
train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(2000))  # Smaller subset for demo
eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(500))

# Step 4: Set up training arguments
training_args = TrainingArguments(
    output_dir="./bert-finetuned-imdb",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Step 5: Define compute_metrics function for evaluation
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    return accuracy

# Step 6: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model
trainer.save_model("./bert-finetuned-imdb-final")
tokenizer.save_pretrained("./bert-finetuned-imdb-final")
print("Model saved to ./bert-finetuned-imdb-final")

# Part 2: Debugging Issues
print("\n=== Part 2: Debugging Issues ===")

# Simulate an issue: Overfitting due to small dataset or high learning rate
# Let's debug by reducing learning rate and adding regularization
training_args.learning_rate = 1e-5  # Lower learning rate
training_args.weight_decay = 0.1    # Increase regularization
training_args.num_train_epochs = 2  # Reduce epochs to prevent overfitting

# Re-initialize trainer with updated args
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Re-train with debugged settings
trainer.train()
print("Debugged training complete. Check logs for improved validation performance.")

# Part 3: Evaluating the Model
print("\n=== Part 3: Evaluating the Model ===")

# Step 1: Generate predictions on test set
test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(500))  # Subset for demo
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = test_dataset["labels"]

# Step 2: Compute evaluation metrics
f1_metric = evaluate.load("f1")
accuracy = accuracy_metric.compute(predictions=preds, references=labels)
f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy['accuracy']:.4f}")
print(f"F1-Score: {f1['f1']:.4f}")

# Step 3: Refine the model (e.g., adjust hyperparameters)
# Example: Increase batch size for stability
training_args.per_device_train_batch_size = 16
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
print("Refined model training complete.")

# Re-evaluate
refined_predictions = trainer.predict(test_dataset)
refined_preds = np.argmax(refined_predictions.predictions, axis=-1)
refined_accuracy = accuracy_metric.compute(predictions=refined_preds, references=labels)
refined_f1 = f1_metric.compute(predictions=refined_preds, references=labels, average="weighted")

print("Refined Evaluation Metrics:")
print(f"Accuracy: {refined_accuracy['accuracy']:.4f}")
print(f"F1-Score: {refined_f1['f1']:.4f}")

# Part 4: Creative Application
print("\n=== Part 4: Creative Application ===")

# Task: Classify short customer reviews as positive/negative
# Simulate a small custom dataset
custom_reviews = [
    {"text": "Great product, fast shipping!", "label": 1},
    {"text": "Terrible quality, waste of money.", "label": 0},
    {"text": "Really happy with my purchase!", "label": 1},
    {"text": "Slow delivery, not impressed.", "label": 0},
]

# Preprocess custom dataset
def preprocess_custom_data(reviews):
    encodings = tokenizer([r["text"] for r in reviews], padding=True, truncation=True, max_length=128, return_tensors="pt")
    encodings["labels"] = torch.tensor([r["label"] for r in reviews])
    return encodings

custom_dataset = preprocess_custom_data(custom_reviews)

# Load fine-tuned model for inference
fine_tuned_model = BertForSequenceClassification.from_pretrained("./bert-finetuned-imdb-final").to(device)
fine_tuned_tokenizer = BertTokenizer.from_pretrained("./bert-finetuned-imdb-final")

# Inference on custom data
fine_tuned_model.eval()
with torch.no_grad():
    inputs = {key: val.to(device) for key, val in custom_dataset.items()}
    outputs = fine_tuned_model(**inputs)
    custom_preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

# Print predictions
print("Creative Application Results:")
for review, pred in zip(custom_reviews, custom_preds):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"Review: {review['text']} -> Predicted Sentiment: {sentiment}")

# Save final model
fine_tuned_model.save_pretrained("./bert-creative-final")
fine_tuned_tokenizer.save_pretrained("./bert-creative-final")
print("Creative application model saved to ./bert-creative-final")
