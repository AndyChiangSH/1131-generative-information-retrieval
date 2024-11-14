import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import argparse


def get_config():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Process input and output file paths.")

    # Add arguments
    parser.add_argument('--config', type=str,
                        required=True, help="name of config")

    # Parse the arguments
    args = parser.parse_args()

    # Load the config file
    with open(os.path.join(f"classifier/config/{args.config}.json"), 'r') as f:
        config = json.load(f)

    return args, config


# Define a custom dataset class
class ClaimDataset(Dataset):
    def __init__(self, claims, tokenizer, max_length=512):
        self.claims = claims
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim_obj = self.claims[idx]
        metadata = claim_obj.get("metadata", {})
        claim = metadata.get("claim", "")
        evidence_sentences = metadata.get("top_relevant_sentences", [])
        # Assuming label is 0 (false), 1 (partial true), or 2 (true)
        label = claim_obj["label"]["rating"]

        # Concatenate claim and evidence sentences
        evidence_text = " ".join(evidence_sentences)
        input_text = f"{claim} {self.tokenizer.sep_token} {evidence_text}"

        # Tokenize the text
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Load claims data
def load_claims(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# Define a function to compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    macro_f1 = f1_score(labels, preds, average='macro')

    return {"macro_f1": macro_f1}


if __name__ == '__main__':
    print("> Start evaluator!")

    # Get configuration
    print("> Get configuration...")
    ARGS, CONFIG = get_config()

    # Set device
    print("> Set device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load validation data
    print("> Load validation data...")
    train_claims = load_claims(os.path.join(CONFIG["input_path"], "train.json"))
    valid_claims = load_claims(os.path.join(CONFIG["input_path"], "valid.json"))


    # Initialize tokenizer and model
    print("> Initialize tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(f"classifier/model/{ARGS.config}/")
    model = BertForSequenceClassification.from_pretrained(
        f"classifier/model/{ARGS.config}/", num_labels=3)
    model.to(device)

    # Create datasets and dataloaders
    print("> Create datasets and dataloaders...")
    train_dataset = ClaimDataset(train_claims, tokenizer)
    valid_dataset = ClaimDataset(valid_claims, tokenizer)

    # Training arguments
    print("> Training arguments...")
    training_args = TrainingArguments(
        output_dir=CONFIG["output_path"],
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1"
    )

    # Define Trainer
    print("> Define Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # Evaluate the model
    print("> Evaluate the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    print("> End evaluator!")
