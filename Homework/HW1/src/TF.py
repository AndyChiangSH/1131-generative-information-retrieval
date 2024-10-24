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


# Implement TF
class TF:
    def __init__(self, corpus):
        self.corpus = [doc.split() for doc in corpus]
        self.vocab = list(set(word for doc in self.corpus for word in doc))
        self.vocab_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.doc_vectors = self._calculate_doc_vectors()

    def _calculate_doc_vectors(self):
        """ Calculate the term frequency (TF) vectors for each document. """
        vectors = np.zeros((len(self.corpus), len(self.vocab)))
        for doc_idx, doc in enumerate(self.corpus):
            for word in doc:
                vectors[doc_idx, self.vocab_index[word]] += 1
        return vectors

    def _calculate_query_vector(self, query):
        """ Calculate the term frequency (TF) vector for a query. """
        query_vector = np.zeros(len(self.vocab))
        for word in query.split():
            if word in self.vocab_index:
                query_vector[self.vocab_index[word]] += 1
        return query_vector

    def cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        return dot_product / (norm_vec1 * norm_vec2)

    def get_top_n(self, query, n=3):
        """ Get the top n documents for a given query using cosine similarity. """
        query_vector = self._calculate_query_vector(query)
        scores = [self.cosine_similarity(
            query_vector, doc_vector) for doc_vector in self.doc_vectors]
        return np.argsort(scores)[-n:][::-1]


if __name__ == '__main__':
    print("START!")

    answer_filename = "TF_1"

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

    # Create TF instance for documents
    print("> Create TF instance for documents...")
    tf = TF(document_texts)

    # Compute similarity for train questions using TF
    print("> Compute similarity for train questions using TF...")
    train_similarity_results_vector = []
    for question in tqdm.tqdm(train_question_texts, desc="Train"):
        top_3_indices = tf.get_top_n(question, n=3)
        top_3_indices = [index + 1 for index in top_3_indices]
        train_similarity_results_vector.append(top_3_indices)

    submission_vector = pd.DataFrame({
        'index': train_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in train_similarity_results_vector]
    })
    submission_vector.to_csv(
        f'answer/train/{answer_filename}.csv', index=False)

    # Calculate Recall@3 for train questions using TF
    print("> Calculate Recall@3 for train questions using TF...")
    train_hits_vector = 0
    for idx, row in train_questions.iterrows():
        true_doc_id = row['Answer ID']
        if true_doc_id in train_similarity_results_vector[idx]:
            train_hits_vector += 1

    train_recall_at_3_vector = train_hits_vector / len(train_questions)
    print(
        f'Recall@3 on train questions using TF: {train_recall_at_3_vector}')

    # Compute similarity for test questions using TF
    print("> Compute similarity for test questions using TF...")
    test_similarity_results_vector = []
    for question in tqdm.tqdm(test_question_texts, desc="Test"):
        top_3_indices = tf.get_top_n(question, n=3)
        top_3_indices = [index + 1 for index in top_3_indices]
        test_similarity_results_vector.append(top_3_indices)

    # Save to CSV for test answers
    print("> Save to CSV for test answers...")
    submission_vector_test = pd.DataFrame({
        'index': test_questions['Question ID'],
        'answer': [" ".join(map(str, indices)) for indices in test_similarity_results_vector]
    })
    submission_vector_test.to_csv(
        f'answer/test/{answer_filename}.csv', index=False)

    print("END!")
