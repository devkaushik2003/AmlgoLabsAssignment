# AMLGOLABS RAG Chatbot

## Project Architecture and Flow

```
Document (PDF/DOCX/TXT) → Preprocessing & Chunking → Embedding & Indexing → Vector DB (FAISS)
      ↓
   User Query → Embedding → Retrieval (Top-k Chunks) → Prompt Construction → Gemini LLM (Streaming) → Response
```

- **Preprocessing**: Extracts and cleans text from documents in `data/`.
- **Chunking**: Splits text into coherent chunks (100–300 words) using NLTK.
- **Embedding**: Uses `all-MiniLM-L6-v2` (SentenceTransformers) to encode chunks.
- **Indexing**: Stores embeddings in a FAISS vector database for fast retrieval.
- **RAG Pipeline**: Retrieves relevant chunks, builds a prompt, and queries Gemini LLM for an answer.
- **Chatbot UI**: Streamlit app for interactive Q&A with source transparency.

## Setup and Running the Pipeline

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Preprocess and Chunk the Document

```bash
python src/preprocess_and_chunk.py
```

- Place your document (PDF, DOCX, or TXT) in the `data/` directory before running.

### 3. Create Embeddings and Build the Vector DB

```bash
python src/embed_and_index.py
```

- This generates `vectordb/faiss.index` and `vectordb/chunk_mapping.json`.

### 4. Run the RAG Chatbot (Streaming Enabled)

```bash
streamlit run app.py
```

- Enter your Gemini API key when prompted.
- Ask questions about the document. Answers are generated with streaming enabled for fast, interactive feedback.

## Model and Embedding Choices

- **Embedding Model**: [`all-MiniLM-L6-v2`](https://www.sbert.net/docs/pretrained_models.html) (SentenceTransformers) for efficient, high-quality semantic search.
- **LLM**: Gemini 2.0 Flash (via Google Generative AI API) for fast, instruction-following responses.
- **Vector DB**: [FAISS](https://faiss.ai/) for scalable, high-speed similarity search.

### Demo Video

- [Watch the demo](https://www.loom.com/share/e41176810e6647ba9a4004a80b7ece6e?sid=9c8c38ce-9a84-
  4a0c-9905-37a81a3fc00a
  )

---

**For more details, see `AMLGOLABS_Project_Report.txt`.**
