# tfidf_cos_dist.py
from sklearn.feature_extraction.text import TfidfVectorizer

def calculate_tfidf_similarity(content1, content2):
    # Use TfidfVectorizer to fit and transform the text documents into TF-IDF vectors
    tfidf = TfidfVectorizer().fit_transform([content1, content2])

    # Calculate pairwise cosine similarity
    pairwise_similarity = (tfidf * tfidf.T).A

    # Extract the similarity value from the matrix
    similarity_percentage = pairwise_similarity[0, 1] * 100

    # Return TF-IDF vectors and similarity percentage
    return tfidf, similarity_percentage
