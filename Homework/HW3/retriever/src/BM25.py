import json
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
import logging

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
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens  # Return tokens for BM25


def calculate_recall_at_k(predictions, true_id, k=30):
    """Calculate if true_id appears in top k predictions"""
    return 1 if true_id in predictions[:k] else 0


def retrieve(train_path, test_path, test_images_path, top_k=30):
    # Load all data
    logging.info("> Loading data...")
    train_data = read_jsonl(train_path)
    test_data = read_jsonl(test_path)
    test_images_data = read_jsonl(test_images_path)
    
    # Extract and preprocess all data
    logging.info("> Extracting and preprocessing data...")
    train_dialogues = [preprocess_text(convert_dialogue_to_text(item["dialogue"])) for item in train_data]
    train_descriptions = [preprocess_text(convert_description_to_text(item["photo_description"])) for item in train_data]
    train_photo_ids = [item["photo_id"] for item in train_data]
    
    test_dialogues = [preprocess_text(convert_dialogue_to_text(item["dialogue"])) for item in test_data]
    test_dialogue_ids = [item["dialogue_id"] for item in test_data]
    test_descriptions = [preprocess_text(convert_description_to_text(item["photo_description"])) for item in test_images_data]
    test_photo_ids = [item["photo_id"] for item in test_images_data]
    
    # Create and train BM25 model
    logging.info("> Training BM25 model...")
    bm25 = BM25Okapi(train_descriptions)
    
    # Evaluate on train data
    logging.info("> Evaluating on train data...")
    train_recall = 0
    for idx, (query, true_photo_id) in enumerate(zip(train_dialogues, train_photo_ids)):
        scores = bm25.get_scores(query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_photo_ids = [train_photo_ids[i] for i in top_indices]
        train_recall += calculate_recall_at_k(top_photo_ids, true_photo_id, top_k)
    
    train_recall_at_k = train_recall / len(train_data)
    logging.info(f"Train Recall@{top_k}: {train_recall_at_k:.4f}")
    
    # Create BM25 for test images
    bm25_test = BM25Okapi(test_descriptions)
    
    # Get test predictions
    logging.info("> Generating test submission...")
    results = []
    for idx, (dialogue_id, query) in enumerate(zip(test_dialogue_ids, test_dialogues)):
        scores = bm25_test.get_scores(query)
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
        "submission_path": "submission/BM25_2.csv",
        "log_path": "log/BM25_2.log"
    }
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(CONFIG["log_path"]),
            logging.StreamHandler()
        ]
    )
    
    logging.info("> Start BM25!")
    logging.info(f"Config: {CONFIG}")
    
    results_df = retrieve(CONFIG["train_path"], CONFIG["test_path"], CONFIG["test_images_path"])
    
    logging.info("> Saving submission...")
    results_df.to_csv(CONFIG["submission_path"], index=False)
    logging.info(f"Created submission with shape: {results_df.shape}")
    
    logging.info("> End BM25!")