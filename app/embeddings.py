from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_embeddings(documents):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc['content'] for doc in documents]
    embeddings = model.encode(texts)
    return embeddings

def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, 'app/vector_store.index')

def load_embeddings():
    index = faiss.read_index('app/vector_store.index')
    return index
