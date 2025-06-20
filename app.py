import streamlit as st
from src.rag_pipeline import rag_answer
import os

# Streamlit provides a simple, interactive web interface for the chatbot.
# This app enables real-time user interaction and displays sources for transparency.

st.set_page_config(page_title="Amlgo Labs RAG Chatbot", layout="wide")

st.title("Amlgo Labs RAG Chatbot")
st.sidebar.header("Model & DB Info")
st.sidebar.write("Model: Gemini 2.0 Flash (via Google API)")
st.sidebar.write("Embedding: all-MiniLM-L6-v2")
st.sidebar.write("Vector DB: FAISS")
st.sidebar.write("Chunks: See /chunks/chunks.jsonl")

api_key = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else st.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("Please enter your Gemini API key to start.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

st.write("Ask a question about the document:")

user_input = st.text_input("Your question", key="user_input")

if st.button("Send") and user_input:
    with st.spinner("Generating answer..."):
        answer, sources = rag_answer(user_input, api_key)
        st.session_state.history.append({"question": user_input, "answer": answer, "sources": sources})

if st.button("Clear Chat"):
    st.session_state.history = []

for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['question']}")
    st.markdown(f"**Bot:** {chat['answer']}")
    with st.expander("Show sources"):
        for i, src in enumerate(chat['sources']):
            st.markdown(f"**Source {i+1}:** {src}")

st.sidebar.write(f"Number of Chunks: {len(os.listdir('chunks')) if os.path.exists('chunks') else 0}") 