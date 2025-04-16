# Retrieval-Augmented-Generation-RAG-System-for-PDF-Document-Interaction-Using-LLMs
The goal of this project is to build a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and The system will leverage LLMs (Gemini) to retrieve and generate accurate, context-aware responses based on the content of the uploaded documents.

## Overview

This Streamlit-based application allows you to **upload multiple PDFs**, **ask questions** about their content using **Retrieval-Augmented Generation (RAG)** using Google's **Gemini 1.5 Flash** model. It uses **FAISS** for semantic search and **SentenceTransformer** embeddings for accurate chunk retrieval.

## Features

- 💬 **Chat Mode:** Ask questions about your uploaded PDFs and get answers generated from relevant document chunks.
- 📂 **Multi-PDF Support:** Upload multiple PDFs and interact with them collectively.
- ⚡ **Fast & Lightweight:** Uses `all-MiniLM-L6-v2` for quick embedding and Gemini Flash for rapid response generation.
- 🧠 **Memory-Aware Chat:** Maintains chat history during session using Streamlit's `session_state`.

## Workflow

1. **PDF Upload:** Upload one or more PDF files.
2. **Text Extraction & Chunking:** Extracts text and splits into manageable chunks.
3. **Embedding:** Converts each chunk into vector embeddings using SentenceTransformers.
4. **Indexing:** Stores embeddings in a FAISS index for fast retrieval.
5. **RAG Pipeline:**
   - On a user question, it retrieves the most relevant chunks.
   - Feeds them to Gemini 1.5 Flash to generate a precise, grounded answer.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- A Google Gemini API key (add it securely via Streamlit secrets)

### Clone the repo

```bash
git clone https://github.com/artikwh/Retrieval-Augmented-Generation-RAG-System-for-PDF-Document-Interaction-Using-LLMs
```

### Install dependencies
```bash
pip install -r requirements.txt
```

🔐 Add Gemini API Key
```bash
gemini_api_key = "your_google_generative_ai_api_key"
```

▶️ Run the App
```bash
streamlit run app.py
```

🧪 Tech Stack

- Streamlit

- Google Generative AI (Gemini 1.5 Flash)

- FAISS (Facebook AI Similarity Search)

- SentenceTransformers

- PyPDF2

🙋‍♀️ Author
Arti Kushwaha
LinkedIn(https://www.linkedin.com/in/arti-kushwaha-32a68634/) | Email(artikwh@gmail.com)

⭐ Feedback or Contributions?
Feel free to open an issue or a pull request. Feedback and improvements are welcome!












