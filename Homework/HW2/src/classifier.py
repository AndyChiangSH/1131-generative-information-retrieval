import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


# Load claims data
def load_claims(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
        top_sentences = metadata.get("top_5_sentences", [])
        label = metadata.get("label", 0)  # Assuming label is 0 (false), 1 (partial true), or 2 (true)
        
        # Concatenate claim and evidence sentences
        evidence_text = " ".join(top_sentences)
        input_text = f"Claim: {claim} Evidence: {evidence_text}"
        
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


if __name__ == '__main__':
    # Define paths
    INPUT_DIR = "retrieval/"
    OUTPUT_DIR = "classifier/"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training and validation data
    train_claims = load_claims(TRAIN_DATA_PATH)
    valid_claims = load_claims(VALID_DATA_PATH)

    # Initialize tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=3)
    model.to(device)

    # Create datasets and dataloaders
    train_dataset = ClaimDataset(train_claims, tokenizer)
    valid_dataset = ClaimDataset(valid_claims, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
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
        metric_for_best_model="f1"
    )

    # Define a function to compute metrics
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        macro_f1 = f1_score(labels, preds, average='macro')
        return {"macro_f1": macro_f1}

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the trained model
    model.save_pretrained(MODEL_OUTPUT_DIR)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

    print(f"Training completed. Model saved in: {MODEL_OUTPUT_DIR}")
