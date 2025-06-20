import os
import json
import re
from nltk.tokenize import sent_tokenize

# Add PDF and DOCX support
import PyPDF2
try:
    import docx
except ImportError:
    docx = None  # python-docx is not in requirements, but we can suggest it if needed

# We use NLTK for robust sentence splitting, which helps in creating meaningful chunks.
# This script prepares the data for embedding and retrieval by cleaning and chunking.

def clean_text(text):
    """
    Remove unwanted characters, headers, footers, and extra spaces from the text.
    """
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    return text.strip()

def chunk_text(text, min_words=100, max_words=300):
    """
    Split text into chunks of 100-300 words, ensuring chunks end at sentence boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        if current_len + len(words) > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(sent)
        current_len += len(words)
        if current_len >= min_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_len = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def read_pdf(file_path):
    """
    Extract text from a PDF file using PyPDF2.
    """
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def read_docx(file_path):
    """
    Extract text from a DOCX file using python-docx.
    """
    if docx is None:
        raise ImportError("python-docx is not installed. Please install it to read .docx files.")
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

def read_txt(file_path):
    """
    Read text from a plain text file.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def main():
    # Check for supported file types in the data directory
    data_dir = 'data'
    output_path = 'chunks/chunks.jsonl'
    os.makedirs('chunks', exist_ok=True)

    file_found = False
    raw_text = ""
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if fname.lower().endswith('.pdf'):
            print(f"Reading from PDF: {fname}")
            raw_text = read_pdf(fpath)
            file_found = True
            break
        elif fname.lower().endswith('.docx'):
            print(f"Reading from DOCX: {fname}")
            raw_text = read_docx(fpath)
            file_found = True
            break
        elif fname.lower().endswith('.txt'):
            print(f"Reading from TXT: {fname}")
            raw_text = read_txt(fpath)
            file_found = True
            break
    if not file_found:
        raise FileNotFoundError("No supported document found in data/ (expected .pdf, .docx, or .txt)")

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            json.dump({'id': i, 'text': chunk}, f)
            f.write('\n')

    print(f"Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    main() 