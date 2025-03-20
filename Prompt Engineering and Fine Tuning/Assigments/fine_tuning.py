# Step 1: Environment Setup
# Install required libraries
# In terminal: pip install torch tensorflow transformers datasets

# Verify GPU availability
import torch
print("GPU Available:", torch.cuda.is_available())  # Should return True if GPU is available

# Step 2: Load the Pre-Trained Model and Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 3: Preprocessing Data
from datasets import load_dataset
dataset = load_dataset("imdb")

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Step 4: Model Training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)

# Fine-tune the model
trainer.train()

# Step 5: Save and Evaluate
# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Evaluate the model
results = trainer.evaluate()
print("Evaluation Results:", results)

# Detailed metrics with sklearn
from sklearn.metrics import classification_report
predictions = trainer.predict(tokenized_dataset["test"])
y_pred = predictions.predictions.argmax(axis=1)
y_true = tokenized_dataset["test"]["label"]
print(classification_report(y_true, y_pred))