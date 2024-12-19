import json
import numpy as np
import pandas as pd
from bert_score import score
import torch
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import logging
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def convert_dialogue_to_text(dialogue):
    """Convert dialogue messages into a single text string"""
    messages = [msg["message"] for msg in dialogue if msg["message"]]
    return " ".join(messages)


def convert_description_to_text(description):
    """Convert description into a single text string"""
    if "Objects in the photo: " in description:
        description = description.replace("Objects in the photo: ", "")
    return description


def preprocess_text(text):
    """Preprocess text with following steps"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    return text  # Return text for BERT


def calculate_recall_at_k(predictions, true_id, k=30):
    """Calculate if true_id appears in top k predictions"""
    return 1 if true_id in predictions[:k] else 0


def retrieve(train_path, test_path, test_images_path, top_k=30):
    # Load all data
    logging.info("> Loading data...")
    train_data = read_jsonl(train_path)
    test_data = read_jsonl(test_path)
    test_images_data = read_jsonl(test_images_path)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Device: {device}")
    
    # Extract and preprocess all data
    logging.info("> Extracting and preprocessing data...")
    train_dialogues = [preprocess_text(convert_dialogue_to_text(item["dialogue"])) for item in train_data]
    train_descriptions = [preprocess_text(convert_description_to_text(item["photo_description"])) for item in train_data]
    train_photo_ids = [item["photo_id"] for item in train_data]
    
    test_dialogues = [preprocess_text(convert_dialogue_to_text(item["dialogue"])) for item in test_data]
    test_dialogue_ids = [item["dialogue_id"] for item in test_data]
    test_descriptions = [preprocess_text(convert_description_to_text(item["photo_description"])) for item in test_images_data]
    test_photo_ids = [item["photo_id"] for item in test_images_data]
    
    # Evaluate on train data
    logging.info("> Evaluating on train data...")
    train_recall = 0
    for idx, (query, true_photo_id) in enumerate(tqdm(zip(train_dialogues, train_photo_ids))):
        _, _, F1 = score([query] * len(train_descriptions), train_descriptions, lang='en', device=device)
        scores = F1.numpy()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_photo_ids = [train_photo_ids[i] for i in top_indices]
        train_recall += calculate_recall_at_k(top_photo_ids, true_photo_id, top_k)
    
    train_recall_at_k = train_recall / len(train_data)
    logging.info(f"Train Recall@{top_k}: {train_recall_at_k:.4f}")
    
    # Get test predictions
    logging.info("> Generating test submission...")
    results = []
    for idx, (dialogue_id, query) in enumerate(tqdm(zip(test_dialogue_ids, test_dialogues))):
        _, _, F1 = score([query] * len(test_descriptions), test_descriptions, lang='en', device=device)
        scores = F1.numpy()
        top_indices = np.argsort(scores)[-top_k:][::-1]
        photo_ids = []
        for photo_idx in top_indices:
            photo_ids.append(test_photo_ids[photo_idx])
            
        results.append({
            'dialogue_id': dialogue_id,
            'photo_id': " ".join(photo_ids)
        })
            
    return pd.DataFrame(results)


if __name__ == "__main__":
    CONFIG = {
        "train_path": "dataset/train.jsonl",
        "test_path": "dataset/test.jsonl",
        "test_images_path": "dataset/test_images.jsonl",
        "submission_path": "submission/BERTScore_1.csv", 
        "log_path": "log/BERTScore_1.log"
    }
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_path"]),
            logging.StreamHandler()
        ]
    )
    
    # Log config and start
    logging.info("> Start BERTScore!")
    logging.info(f"Config: {CONFIG}")
    
    # Get results with train evaluation 
    results_df = retrieve(CONFIG["train_path"], CONFIG["test_path"], CONFIG["test_images_path"])
    
    # Save submission
    logging.info("> Saving submission...")
    results_df.to_csv(CONFIG["submission_path"], index=False)
    logging.info(f"Created submission with shape: {results_df.shape}")
    
    logging.info("> End BERTScore!")