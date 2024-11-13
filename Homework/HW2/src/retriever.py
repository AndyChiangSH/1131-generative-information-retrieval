import os
import json
import re
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from bert_score import score
import torch


# Preprocess claim or sentence text
def preprocess_text(text):
    # 1. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 2. Convert to lowercase
    text = text.lower()

    # 3. Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text


# Retrieve the top 5 relevant sentences for each claim
def retrieve_top_sentences(claim, premise_articles):
    sentences = []
    for url, article_name in premise_articles.items():
        article_path = os.path.join(ARTICLES_DIR, article_name)
        try:
            with open(article_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)
                if isinstance(article_data, list):
                    sentences.extend(article_data)
        except Exception as e:
            print(f"Error loading {article_path}: {e}")
    
    # Preprocess claim and sentences
    preprocessed_claim = preprocess_text(claim)
    if IS_UNIQUE:
        preprocessed_sentences = list(set(preprocess_text(sentence) for sentence in sentences))
    else:
        preprocessed_sentences = list(preprocess_text(sentence) for sentence in sentences)
    
    # print("sentences:", sentences)
    # print("preprocessed_sentences:", preprocessed_sentences)
    
    if len(preprocessed_claim) <= 0 or len(preprocessed_sentences) <= 0:
        return []
    
    if IR == "TF-IDF":
        # Create TF-IDF representation
        vectorizer = TfidfVectorizer(stop_words='english')
        # Fit the vectorizer on the sentences and transform them
        tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
        # Transform the claim into the TF-IDF space
        claim_vector = vectorizer.transform([preprocessed_claim])
        # Calculate cosine similarity between the claim and all sentences
        cosine_similarities = cosine_similarity(claim_vector, tfidf_matrix).flatten()
        # Get the top most relevant sentences
        top_indices = np.argsort(cosine_similarities)[::-1][:TOP_K]
    elif IR == "BM25":
        # Tokenize sentences for BM25
        tokenized_sentences = [sentence.split() for sentence in preprocessed_sentences]
        # Initialize BM25 model
        bm25 = BM25Okapi(tokenized_sentences)
        # Tokenize the claim
        tokenized_claim = preprocessed_claim.split()
        # Get BM25 scores for all sentences
        bm25_scores = bm25.get_scores(tokenized_claim)
        # Get the top most relevant sentences
        top_indices = np.argsort(bm25_scores)[::-1][:TOP_K]
    elif IR == "BERTScore":
        # Calculate BERTScore between the claim and all sentences
        P, R, F1 = score(preprocessed_sentences, [preprocessed_claim] * len(preprocessed_sentences), lang='en', device='cuda' if torch.cuda.is_available() else 'cpu')
        # Get the top 5 most relevant sentences based on BERTScore F1
        f1_scores = F1.tolist()
        top_indices = sorted(range(len(f1_scores)), key=lambda i: f1_scores[i], reverse=True)[:TOP_K]
    else:
        print("Invalid IR algorithm. Please choose between 'TF-IDF' and 'BM25'.")
        return []
    
    # Return the top relevant sentences
    top_sentences = [preprocessed_sentences[idx] for idx in top_indices]
    
    return top_sentences


# Process each claim and add top relevant sentences
def retrieve(input_path, output_path):
    # Load the input data
    print(f">> Load the input data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
        
    # Process each claim and add top relevant sentences to the metadata of the claim
    print(">> Retrieve top sentences...")
    for data in tqdm(datas):
        # print("data:", data)
        
        metadata = data.get("metadata", {})
        claim = metadata.get("claim", "")
        premise_articles = metadata.get("premise_articles", {})
        if claim and premise_articles:
            top_relevant_sentences = retrieve_top_sentences(claim, premise_articles)
            # Add the top relevant sentences to the claim metadata
            metadata["top_relevant_sentences"] = top_relevant_sentences
        else:
            metadata["top_relevant_sentences"] = []
    
    # Save the updated claims to the output file
    print(f"Save the output data to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(datas, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    print("> Start retriever!")
    
    # Define directories
    ARTICLES_DIR = "dataset/articles/"
    INPUT_DIR = "dataset/"
    OUTPUT_DIR = "retriever/BERTScore/top-10-unique/"
    IR = "BERTScore"
    TOP_K = 10
    IS_UNIQUE = True
    
    # Download nltk toolkit
    print("> Download nltk toolkit...")
    nltk.download('stopwords')

    # Create output directory if it does not exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Process train, valid, and test claims and save them
    print("> Retrieve for train.json...")
    retrieve(os.path.join(INPUT_DIR, "train.json"), os.path.join(OUTPUT_DIR, "train.json"))
    print("> Retrieve for valid.json...")
    retrieve(os.path.join(INPUT_DIR, "valid.json"), os.path.join(OUTPUT_DIR, "valid.json"))
    print("> Retrieve for test.json...")
    retrieve(os.path.join(INPUT_DIR, "test.json"), os.path.join(OUTPUT_DIR, "test.json"))

    print("> End retriever!")
