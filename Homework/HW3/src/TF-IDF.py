import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def calculate_recall_at_k(predictions, true_id, k=30):
    """Calculate if true_id appears in top k predictions"""
    return 1 if true_id in predictions[:k] else 0

def evaluate_and_retrieve(train_path, test_path, test_images_path, top_k=30):
    # Load all data
    print("> Loading data...")
    train_data = read_jsonl(train_path)
    test_data = read_jsonl(test_path)
    test_images_data = read_jsonl(test_images_path)
    
    # Extract all data
    print("> Extracting data...")
    train_dialogues = [convert_dialogue_to_text(item["dialogue"]) for item in train_data]
    train_descriptions = [item["photo_description"] for item in train_data]
    train_photo_ids = [item["photo_id"] for item in train_data]
    
    test_dialogues = [convert_dialogue_to_text(item["dialogue"]) for item in test_data]
    test_dialogue_ids = [item["dialogue_id"] for item in test_data]
    test_descriptions = [item["photo_description"] for item in test_images_data]
    test_photo_ids = [item["photo_id"] for item in test_images_data]
    
    # Create and train TF-IDF vectorizer
    print("> Training TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(train_descriptions + train_dialogues)
    
    # Evaluate on train data
    print("> Evaluating on train data...")
    train_dialogue_vectors = vectorizer.transform(train_dialogues)
    train_photo_vectors = vectorizer.transform(train_descriptions)
    
    train_recall = 0
    for idx, true_photo_id in enumerate(train_photo_ids):
        scores = cosine_similarity(train_dialogue_vectors[idx:idx+1], train_photo_vectors)[0]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_photo_ids = [train_photo_ids[i] for i in top_indices]
        train_recall += calculate_recall_at_k(top_photo_ids, true_photo_id, top_k)
    
    train_recall_at_k = train_recall / len(train_data)
    print(f"Train Recall@{top_k}: {train_recall_at_k:.4f}")
    
    # Get test predictions
    print("> Generating test submission...")
    dialogue_vectors = vectorizer.transform(test_dialogues)
    photo_vectors = vectorizer.transform(test_descriptions)
    
    results = []
    for idx, dialogue_id in enumerate(test_dialogue_ids):
        scores = cosine_similarity(dialogue_vectors[idx:idx+1], photo_vectors)[0]
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
    train_path = "dataset/train.jsonl"
    test_path = "dataset/test.jsonl"
    test_images_path = "dataset/test_images.jsonl"
    submission_path = "submission/TF-IDF_1.csv"
    
    # Get results with train evaluation
    results_df = evaluate_and_retrieve(train_path, test_path, test_images_path)
    
    # Save submission
    print("> Saving submission...")
    results_df.to_csv(submission_path, index=False)
    print(f"Created submission with shape: {results_df.shape}")