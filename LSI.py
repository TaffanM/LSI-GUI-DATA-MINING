# lsi_module.py
import os
from collections import Counter
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from docx import Document
import PyPDF2
import string

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in nltk.corpus.stopwords.words('indonesian')]
    return ' '.join(words)

def read_text_document(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx_document(filepath):
    doc = Document(filepath)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

def read_pdf_document(filepath):
    with open(filepath, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def read_document(filepath):
    _, ext = os.path.splitext(filepath.lower())
    if ext == '.txt':
        return read_text_document(filepath)
    elif ext == '.docx':
        return read_docx_document(filepath)
    elif ext == '.pdf':
        return read_pdf_document(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def stemming_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def list_files_dir(directory):
    files = os.listdir(directory)
    return files

def perform_lsi(query, directory):
    documents = [read_document(os.path.join(directory, filename)) for filename in os.listdir(directory)]
    preprocessed_documents = [stemming_text(preprocess_text(doc)) for doc in documents]
    preprocessed_query = stemming_text(preprocess_text(query))

    bow_vectorizer = CountVectorizer()
    bow_matrix = bow_vectorizer.fit_transform(preprocessed_documents)

    num_topics = 2
    svd_model = TruncatedSVD(n_components=num_topics)
    lsa_topic_matrix = svd_model.fit_transform(bow_matrix)

    query_vector = bow_vectorizer.transform([preprocessed_query])
    query_topic = svd_model.transform(query_vector)

    cosine_similarity_scores = cosine_similarity(query_topic, lsa_topic_matrix)
    ranking = np.argsort(cosine_similarity_scores[0])[::-1]

    results = []
    for rank, doc_index in enumerate(ranking):
        doc_filename = os.listdir(directory)[doc_index]
        similarity_score = cosine_similarity_scores[0][doc_index]
        results.append((rank + 1, doc_filename, similarity_score))

    return results

if __name__ == "__main__":
    # Example usage
    query = "Nasi goreng seafood dan bolognese disertai caesar salad"
    directory = "infoRetrieval/"
    results = perform_lsi(query, directory)

    print("Dokumen yang paling relevan:")
    for rank, doc_filename, similarity_score in results:
        print("Rank: {}, Dokumen: {}, Cosine Similarity: {}".format(rank, doc_filename, similarity_score))
