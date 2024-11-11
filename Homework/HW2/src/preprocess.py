import os
import json
import re
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords


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


def preprocess_article(file_path, save_path):
    """
    Preprocess a single article file.
    """
    try:
        # Load the article JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            article_words = json.load(f)
        
        # Clean the word
        preprocess_words = []
        for word in article_words:
            preprocess_word = preprocess_text(word)
            
            if len(preprocess_word) > 0:
                preprocess_words.append(preprocess_word)
            
        # Save the cleaned content to the new folder
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(preprocess_words, f, ensure_ascii=False)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    print("> Start preprocessing!")
    
    # Define directories
    RAW_ARTICLES_DIR = "dataset/articles/"
    PROCESSED_ARTICLES_DIR = "dataset/preprocessed_articles/"
    
    # Download nltk toolkit
    print("> Download nltk toolkit...")
    nltk.download('stopwords')
    
    print(f"> Read articles from {RAW_ARTICLES_DIR}...")

    # Create a new directory for processed articles if it does not exist
    if not os.path.exists(PROCESSED_ARTICLES_DIR):
        os.makedirs(PROCESSED_ARTICLES_DIR)

    # Iterate over all articles in the raw articles directory
    for article_filename in tqdm(os.listdir(RAW_ARTICLES_DIR)):
        raw_article_path = os.path.join(RAW_ARTICLES_DIR, article_filename)
        processed_article_path = os.path.join(PROCESSED_ARTICLES_DIR, article_filename)
        
        # Preprocess and save each article
        preprocess_article(raw_article_path, processed_article_path)

    print(f"> Save preprocessed articles in {PROCESSED_ARTICLES_DIR}...")
    
    print("> End preprocessing!")
