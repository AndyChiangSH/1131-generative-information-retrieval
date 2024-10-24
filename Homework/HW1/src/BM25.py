import numpy as np
import pandas as pd
import tqdm
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


# Implement BM25
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = [doc.split() for doc in corpus]
        self.k1 = k1
        self.b = b
        self.doc_lengths = np.array([len(doc) for doc in self.corpus])
        self.avg_doc_length = np.mean(self.doc_lengths)
        self.vocab = list(set(word for doc in self.corpus for word in doc))
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.doc_freqs = self._calculate_doc_freqs()
        self.idf_cache = self._calculate_idfs()

    def _calculate_doc_freqs(self):
        """ Calculate document frequencies for each term in the vocabulary. """
        df = np.zeros(len(self.vocab))
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                term_idx = self.vocab_index[term]
                df[term_idx] += 1
                
        return df

    def _calculate_idfs(self):
        """ Precompute IDF values for each term in the vocabulary. """
        n_docs = len(self.corpus)
        idf = np.log((n_docs - self.doc_freqs + 0.5) /
                     (self.doc_freqs + 0.5) + 1)
        
        return idf

    def score(self, query, document_idx):
        """ Score a document for a given query using BM25. """
        score = 0
        document = self.corpus[document_idx]
        doc_length = self.doc_lengths[document_idx]
        for term in query:
            if term in self.vocab_index:
                term_idx = self.vocab_index[term]
                tf = document.count(term)
                idf = self.idf_cache[term_idx]
                denom = tf + self.k1 * \
                    (1 - self.b + self.b * (doc_length / self.avg_doc_length))
                score += idf * ((tf * (self.k1 + 1)) / denom)
                
        return score

    def get_top_n(self, query, n=3):
        """ Get the top n documents for a given query. """
        query_terms = query.split()
        scores = np.array([self.score(query_terms, doc_idx)
                          for doc_idx in range(len(self.corpus))])
        
        return np.argsort(scores)[-n:][::-1]


if __name__ == '__main__':
    print("START!")

    answer_filename = "BM25_5"

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

    # Create BM25 instance for documents
    print("> Create BM25 instance for documents...")
    bm25 = BM25(document_texts, k1=1.75, b=1.0)

    # Compute similarity for train questions using BM25
    print("> Compute similarity for train questions using BM25...")
    train_similarity_results_bm25 = []
    for question in tqdm.tqdm(train_question_texts, desc="Train"):
        top_3_indices = bm25.get_top_n(question, n=3)
        top_3_indices = [index + 1 for index in top_3_indices]
        train_similarity_results_bm25.append(top_3_indices)

    # Save to CSV for train answers
    print("> Save to CSV for train answers...")
    submission_bm25 = pd.DataFrame({
        'index': train_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in train_similarity_results_bm25]
    })
    submission_bm25.to_csv(
        f'answer/train/{answer_filename}.csv', index=False)

    # Calculate Recall@3 for train questions using BM25
    print("> Calculate Recall@3 for train questions using BM25...")
    train_hits_bm25 = 0
    for idx, row in train_questions.iterrows():
        true_doc_id = row['Answer ID']
        if true_doc_id in train_similarity_results_bm25[idx]:
            train_hits_bm25 += 1

    train_recall_at_3_bm25 = train_hits_bm25 / len(train_questions)
    print(f'Recall@3 on train questions using BM25: {train_recall_at_3_bm25}')

    # Compute similarity for test questions using BM25
    print("> Compute similarity for test questions using BM25...")
    test_similarity_results_bm25 = []
    for question in tqdm.tqdm(test_question_texts, desc="Test"):
        top_3_indices = bm25.get_top_n(question, n=3)
        top_3_indices = [index + 1 for index in top_3_indices]
        test_similarity_results_bm25.append(top_3_indices)

    # Save to CSV for test answers
    print("> Save to CSV for test answers...")
    submission_bm25_test = pd.DataFrame({
        'index': test_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in test_similarity_results_bm25]
    })
    submission_bm25_test.to_csv(
        f'answer/test/{answer_filename}.csv', index=False)

    print("END!")
