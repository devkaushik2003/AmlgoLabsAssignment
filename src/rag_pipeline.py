import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# This script implements the core RAG logic: retrieve relevant chunks and generate an answer using Gemini.

def load_faiss_index(vectordb_dir):
    index = faiss.read_index(os.path.join(vectordb_dir, 'faiss.index'))
    with open(os.path.join(vectordb_dir, 'chunk_mapping.json'), 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    return index, mapping

def retrieve(query, model, index, mapping, top_k=3):
    """
    Retrieve top-k most relevant chunks for the query.
    """
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        results.append(mapping['texts'][idx])
    return results

def build_prompt(query, retrieved_chunks):
    """
    Build a prompt for Gemini with retrieved context and user query.
    """
    context = "\n\n".join([f"Source {i+1}: {chunk}" for i, chunk in enumerate(retrieved_chunks)])
    prompt = (
        f"You are an AI assistant. Use ONLY the following sources to answer the user's question.\n\n"
        f"{context}\n\n"
        f"User question: {query}\n\n"
        f"Answer (cite sources as Source 1, Source 2, etc. if used):"
    )
    return prompt

def generate_answer(prompt, api_key):
    """
    Call Gemini API to generate an answer.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    # We use the Gemini API for high-quality, instruction-following generation.
    response = model.generate_content(prompt, stream=False)
    return response.text

def rag_answer(query, api_key, vectordb_dir='vectordb', top_k=3):
    # Load embedding model and vector DB
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    index, mapping = load_faiss_index(vectordb_dir)
    # Retrieve relevant chunks
    retrieved = retrieve(query, embed_model, index, mapping, top_k=top_k)
    # Build prompt and generate answer
    prompt = build_prompt(query, retrieved)
    answer = generate_answer(prompt, api_key)
    return answer, retrieved 