import os
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


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


# Load valid claims data
def load_claims(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
        claim_id = metadata.get("id", "")
        evidence_sentences = metadata.get("top_relevant_sentences", [])
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
            'claim_id': claim_id,
            'label': label
        }


if __name__ == '__main__':
    print("> Start evaluator!")

    # Get configuration
    print("> Get configuration...")
    ARGS, CONFIG = get_config()

    # Set device
    print("> Set device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load trained model and tokenizer
    print("> Load trained model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["output_path"])
    model = AutoModelForSequenceClassification.from_pretrained(CONFIG["output_path"])
    model.to(device)
    model.eval()

    # Load valid claims
    print("> Load valid claims...")
    valid_claims = load_claims(os.path.join(CONFIG["input_path"], "valid.json"))

    # Create dataset and dataloader
    print("> Create dataset and dataloader...")
    valid_dataset = ClaimDataset(valid_claims, tokenizer)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    # Prepare submission list
    submission_data = []

    # Predict labels for the valid data
    print("> Predict labels for the valid data...")
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            claim_ids = batch['claim_id']
            labels = batch['label']
            
            # Get predictions
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Append predictions to the submission list
            for claim_id, label, pred in zip(claim_ids, labels, preds):
                submission_data.append({"id": int(claim_id), "label": int(label), "pred": int(pred)})

    # Create output directory if it does not exist
    if not os.path.exists(f"evaluator/{ARGS.config}/"):
        os.makedirs(f"evaluator/{ARGS.config}/")
    
    # Create a DataFrame and save to CSV
    print("> Create a DataFrame and save to CSV...")
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(f"evaluator/{ARGS.config}/result.csv", index=False)
    
    # Generate confusion matrix
    print("> Generate confusion matrix...")
    label_names = ['False', 'Partial True', 'True']
    cm = confusion_matrix(y_true=submission_df["label"], y_pred=submission_df["pred"], labels=[0, 1, 2])
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(f"evaluator/{ARGS.config}/confusion_matrix.jpg")
    plt.show()
    
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Convert to percentages
    sns.heatmap(cm_percentage, annot=True, fmt='.2f', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Percentage)')
    plt.savefig(f"evaluator/{ARGS.config}/confusion_matrix_percentage.jpg")
    plt.show()

    print("> End evaluator!")
