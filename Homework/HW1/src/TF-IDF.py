import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer


# Define a function for text preprocessing
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

    # 5. Stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    # 6. Lemmatization
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    # 7. Remove numbers
    text = re.sub(r'\d+', '', text)

    # 8. Remove extra spaces
    text = ' '.join(text.split())

    return text


# Compute TF-IDF of the corpus
def compute_tfidf(corpus, vocabulary=None):
    # Tokenize the corpus
    print(">> Tokenize the corpus...")
    tokenized_corpus = [doc.split() for doc in corpus]
    
    # Create the vocabulary index
    print(">> Create the vocabulary index...")
    vocab_index = {word: idx for idx, word in enumerate(vocabulary)}

    # Calculate term frequency (TF)
    print(">> Calculate term frequency (TF)...")
    tf = np.zeros((len(corpus), len(vocabulary)))
    for doc_idx, doc in enumerate(tokenized_corpus):
        for word in doc:
            tf[doc_idx, vocab_index[word]] += 1
        tf[doc_idx] /= len(doc)

    # Calculate document frequency (DF)
    print(">> Calculate document frequency (DF)...")
    # df = np.zeros(len(vocabulary))
    # for word in tqdm.tqdm(vocabulary):
    #     df[vocab_index[word]] = sum(
    #         1 for doc in tokenized_corpus if word in doc)
    
    # Convert tokenized corpus to a document-term matrix (boolean representation)
    dtm = np.zeros((len(tokenized_corpus), len(vocabulary)), dtype=np.int32)
    for doc_idx, doc in enumerate(tokenized_corpus):
        for word in set(doc):  # Use set to avoid redundant counting
            dtm[doc_idx, vocab_index[word]] = 1

    # Calculate document frequency (DF) using the document-term matrix
    df = np.sum(dtm, axis=0)

    # Calculate inverse document frequency (IDF)
    print(">> Calculate inverse document frequency (IDF)...")
    idf = np.log(len(corpus) / (df + 1))

    # Calculate TF-IDF
    print(">> Calculate TF-IDF...")
    tfidf = tf * idf

    return tfidf


# Compute cosine similarity between each question and all documents
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    return dot_product / (norm_vec1 * norm_vec2)


if __name__ == '__main__':
    print("START!")
    
    answer_filename = "TF-IDF_1"
    
    # Download nltk toolkit
    print("> Download nltk toolkit...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    
    # Load the datasets
    print("> Load the datasets...")
    documents_data = pd.read_csv('dataset/documents_data.csv')
    train_questions = pd.read_csv('dataset/train_question.csv')
    test_questions = pd.read_csv('dataset/test_question.csv')

    # Extract document texts and question texts
    document_texts = documents_data['Document_HTML']
    train_question_texts = train_questions['Question']
    test_question_texts = test_questions['Question']
    
    # Apply preprocessing to all texts
    print("> Apply preprocessing to all texts...")
    document_texts = document_texts.apply(preprocess_text)
    train_question_texts = train_question_texts.apply(preprocess_text)
    test_question_texts = test_question_texts.apply(preprocess_text)
    
    # Combine all texts to create a shared vocabulary
    print("> Combine all texts to create a shared vocabulary...")
    all_texts = pd.concat([document_texts, train_question_texts, test_question_texts])
    tokenized_corpus = [doc.split() for doc in all_texts]
    vocabulary = list(set(word for doc in tokenized_corpus for word in doc))
    
    # Compute TF-IDF vectors for documents
    print("> Compute TF-IDF vectors for documents...")
    document_tfidf = compute_tfidf(document_texts, vocabulary=vocabulary)
    
    # Compute TF-IDF vectors for train questions
    print("> Compute TF-IDF vectors for train questions...")
    train_question_tfidf = compute_tfidf(train_question_texts, vocabulary=vocabulary)
    
    # Compute TF-IDF vectors for test questions
    print("> Compute TF-IDF vectors for test questions...")
    test_question_tfidf = compute_tfidf(test_question_texts, vocabulary=vocabulary)

    # Compute similarity for train questions
    print("> Compute similarity for train questions...")
    train_similarity_results = []
    for question_vector in train_question_tfidf:
        cosine_similarities = [cosine_similarity(
            question_vector, doc_vector) for doc_vector in document_tfidf]
        # Get indices of top 3 similar documents
        top_3_indices = np.argsort(cosine_similarities)[-3:][::-1]
        top_3_indices = [index + 1 for index in top_3_indices]
        train_similarity_results.append(top_3_indices)

    # Save to CSV for train answers
    print("> Save to CSV for train answers...")
    submission = pd.DataFrame({
        'index': train_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in train_similarity_results]
    })
    submission.to_csv(f'answer/train/{answer_filename}.csv', index=False)

    # Calculate Recall@3 for train questions
    print("> Calculate Recall@3 for train questions...")
    train_hits = 0
    for idx, row in train_questions.iterrows():
        true_doc_id = row['Answer ID']
        if true_doc_id in train_similarity_results[idx]:
            train_hits += 1

    train_recall_at_3 = train_hits / len(train_questions)
    print(f'Recall@3 on train questions: {train_recall_at_3}')

    # Compute similarity for test questions
    print("> Compute similarity for test questions...")
    test_similarity_results = []
    for question_vector in test_question_tfidf:
        cosine_similarities = [cosine_similarity(
            question_vector, doc_vector) for doc_vector in document_tfidf]
        # Get indices of top 3 similar documents
        top_3_indices = np.argsort(cosine_similarities)[-3:][::-1]
        top_3_indices = [index + 1 for index in top_3_indices]
        test_similarity_results.append(top_3_indices)

    # Save to CSV for test answers
    print("> Save to CSV for test answers...")
    submission = pd.DataFrame({
        'index': test_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in test_similarity_results]
    })
    submission.to_csv(f'answer/test/{answer_filename}.csv', index=False)
    
    print("END!")
