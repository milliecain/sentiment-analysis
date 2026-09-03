import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(
    "sentiment_scored.csv",
    encoding="latin1"
)

df["sentiment"] = pd.to_numeric(
    df["sentiment"],
    errors="coerce"
)

df = df.dropna(subset=["sentiment"])

print("Examples:", len(df))

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Train:", len(train_df))
print("Test:", len(test_df))

# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = Dataset.from_pandas(
    train_df[["def", "sentiment"]]
)

test_dataset = Dataset.from_pandas(
    test_df[["def", "sentiment"]]
)

# --------------------------------------------------
# TOKENS
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

def tokenize(batch):

    return tokenizer(
        batch["def"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)

# --------------------------------------------------
# LABELS
# --------------------------------------------------

def rename_labels(example):

    example["labels"] = float(
        example["sentiment"]
    )

    return example

train_dataset = train_dataset.map(
    rename_labels
)

test_dataset = test_dataset.map(
    rename_labels
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=1,
    problem_type="regression"
)

# --------------------------------------------------
# METRICS
# --------------------------------------------------

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.reshape(-1)

    mae = np.mean(
        np.abs(predictions - labels)
    )

    return {"mae": mae}

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

results = trainer.evaluate()

print(results)

