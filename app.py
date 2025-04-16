#importing the required libraries
import google.generativeai as genai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
import streamlit as st
import os

st.set_page_config(page_title="RAG System", page_icon="🤖")

# The API key is used to authenticate requests to the Google Generative AI service.
api_key = "your_api_key_here"  # Replace with your actual API key

#connecting api key to the google generative ai
genai.configure(api_key=api_key)

#model selection
model = genai.GenerativeModel("gemini-1.5-flash")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Session state setup
if "messages" not in st.session_state:
    st.session_state.messages = []

if "faiss_index" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.faiss_index = None

def chunk_text(text, chunk_size=500):
    """Splits the text into smaller chunks of specified size."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(text), chunk_size)]

def vector_search(query, top_val = 3):
  """Searches for the most relevant text chunks based on the query using vector embeddings."""
  query_emb = np.array([embedding_model.encode(query)]).astype('float32')

  v, ind = index.search(query_emb, top_val)

  return " ".join([chunks[i] for i in ind[0]])

def get_answer(question):
  """Generates an answer to the question using the relevant text from the PDF document."""
  relev_text = vector_search(question)

  prompt_text = f"""

  You are a AI assistant /  
  I have a PDF document, and I will ask questions based on its content /
  if user is asking any short form of questions, you have to take a details about it and find the answer /
  Answer the questions by using PDF content /
  You are not allowed to answer any questions that are not related to the PDF content /
  you can answer in a single sentence using maximum 30 words/
  Do not make over information /

  Here is the PDF content:

  {relev_text}

  Question: {question}

  """

  model = genai.GenerativeModel("gemini-1.5-flash")

  resp = model.generate_content(prompt_text)

  return resp.text

#streamlit page title
st.header("Retrieval-Augmented Generation (RAG) System for PDF Document Interaction" "🤖")
st.text("This application allows you to ask questions about the content of a uploaded PDF document. The system retrieves relevant text from the document and generates answers using a generative AI model.")

pdf_files = st.file_uploader("Upload PDF Files", type= "pdf", accept_multiple_files = True)

if pdf_files:
    st.write("PDF uploaded successfully.")
    text = ""
    # Loop through the uploaded PDF files and extract text
    for pdf_file in pdf_files:        
        # Read the PDF file and extract text
        doc = PyPDF2.PdfReader(pdf_file)
        st.write("Processing PDF file:", pdf_file.name)
        # Extract text from each page and concatenate
        for page_num in range(len(doc.pages)):
            text += doc.pages[page_num].extract_text()
        
    #st.write("Text extracted from PDF:", text)

    # Chunk the text into smaller pieces for processing
    chunks = chunk_text(text)
    #st.write("Chunks of text:", chunks)
    
    # Create embeddings for the chunks
    chunk_embeddings = [embedding_model.encode(i) for i in chunks]
    embeddings = np.array(chunk_embeddings).astype("float32")
    #st.write("Embeddings created for the chunks.", embeddings.shape)

    # Create a FAISS index for the embeddings
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    # Save in session state
    st.session_state.chunks = chunks
    st.session_state.faiss_index = index

# Create full-width button using columns

if pdf_files:
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:    
        if st.tabs(["❓ Ask a Question about the PDF document."]):
                        
            # User input for query
            query = st.text_input("Enter your question:")
            
            if query:
                # Generate an answer to the question using the relevant text from the PDF document
                answer = get_answer(query)
                st.session_state.messages.append({"role": "user", "text": query})
                st.session_state.messages.append({"role": "ai", "text": answer})

            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f"🧑‍💻 **You:** {msg['text']}")
                else:
                    st.markdown(f"🤖 **Gemini:** {msg['text']}")
        
