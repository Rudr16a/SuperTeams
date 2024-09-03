# Example code for data preprocessing
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

# Load the dataset
df = pd.read_csv('medical_data.csv')

# Preprocess text for sparse vector generation
vectorizer = TfidfVectorizer(stop_words='english')
sparse_vectors = vectorizer.fit_transform(df['text'])

# Generate dense vectors using BioBERT
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')
model = AutoModel.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

def generate_dense_vector(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

df['dense_vectors'] = df['text'].apply(generate_dense_vector)

# Indexing sparse vectors in Qdrant
from qdrant_client import QdrantClient

client = QdrantClient()

# Create a collection in Qdrant for sparse vectors
client.create_collection('medical_sparse_vectors', vector_size=sparse_vectors.shape[1])

# Indexing the sparse vectors
client.upload_collection(
    collection_name='medical_sparse_vectors',
    vectors=sparse_vectors.toarray(),
    payload=df['text'].tolist()
)

# Querying sparse vectors
query = vectorizer.transform(['diabetes treatment']).toarray()
results = client.search(
    collection_name='medical_sparse_vectors',
    query_vector=query[0],
    top=5
)

# Indexing dense vectors in Qdrant
client.create_collection('medical_dense_vectors', vector_size=768)

# Indexing the dense vectors
client.upload_collection(
    collection_name='medical_dense_vectors',
    vectors=df['dense_vectors'].tolist(),
    payload=df['text'].tolist()
)

# Querying dense vectors
query_vector = generate_dense_vector('treatment for high blood sugar')
results = client.search(
    collection_name='medical_dense_vectors',
    query_vector=query_vector,
    top=5
)
# Hybrid search combining sparse and dense vectors
sparse_query = vectorizer.transform(['diabetes']).toarray()
dense_query = generate_dense_vector('treatment for high blood sugar')

sparse_results = client.search(
    collection_name='medical_sparse_vectors',
    query_vector=sparse_query[0],
    top=5
)

dense_results = client.search(
    collection_name='medical_dense_vectors',
    query_vector=dense_query,
    top=5
)

# Combine results using a custom weighting mechanism
combined_results = combine_results(sparse_results, dense_results)

import streamlit as st

def search(query):
    # Generate sparse vector and search
    sparse_query = vectorizer.transform([query]).toarray()
    sparse_results = client.search(
        collection_name='medical_sparse_vectors',
        query_vector=sparse_query[0],
        top=5
    )
    
    # Generate dense vector and search
    dense_query = generate_dense_vector(query)
    dense_results = client.search(
        collection_name='medical_dense_vectors',
        query_vector=dense_query,
        top=5
    )
    
    # Combine results using a custom weighting mechanism
    combined_results = combine_results(sparse_results, dense_results)
    
    return sparse_results, dense_results, combined_results

# Streamlit UI
st.title("Hybrid Search System for Medical Domain")
query = st.text_input("Enter your search query:")

if query:
    sparse_results, dense_results, combined_results = search(query)
    
    st.subheader("Results using Sparse Vectors:")
    for result in sparse_results:
        st.write(result['payload'])
    
    st.subheader("Results using Dense Vectors:")
    for result in dense_results:
        st.write(result['payload'])
    
    st.subheader("Combined Hybrid Results:")
    for result in combined_results:
        st.write(result['payload'])

