import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# SentenceTransformer provides high-quality semantic embeddings for text.
# FAISS is a fast, efficient vector database for similarity search.

def load_chunks(chunk_path):
    """
    Load text chunks from a JSONL file.
    """
    chunks = []
    with open(chunk_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def main():
    chunk_path = 'chunks/chunks.jsonl'
    vectordb_dir = 'vectordb'
    os.makedirs(vectordb_dir, exist_ok=True)

    # Load pre-trained embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    chunks = load_chunks(chunk_path)
    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index and mapping
    faiss.write_index(index, os.path.join(vectordb_dir, 'faiss.index'))
    with open(os.path.join(vectordb_dir, 'chunk_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump({'ids': ids, 'texts': texts}, f)

    print(f"Saved FAISS index and mapping to {vectordb_dir}")

if __name__ == "__main__":
    main() 