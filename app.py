import streamlit as st
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader
import os
import numpy as np

# Streamlit UI
st.title("AI Clone Chatbot")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Load and process PDF
def load_pdf(file):
    """Load and extract text from a PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if uploaded_file:
    pdf_text = load_pdf(uploaded_file)
    st.success("PDF successfully loaded!")

# Chunk text
def chunk_text(text):
    """Split the extracted text into smaller chunks."""
    chunk_size = 600
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return splitter.split_text(text)

if uploaded_file:
    chunks = chunk_text(pdf_text)
    st.write(f"Text split into {len(chunks)} chunks.")

# Initialize ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Store embeddings
def store_embeddings(chunks, collection, embedding_model):
    """Embed and store new text chunks in ChromaDB."""
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )

if uploaded_file:
    store_embeddings(chunks, collection, embedding_model)
    st.success("Embeddings stored in ChromaDB!")

# Chat Model
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="your-api-key")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Query function
def query_llama3(user_query):
    """Handles user queries and retrieves context."""
    retrieved_context = collection.query(query_embeddings=[embedding_model.embed_query(user_query)], n_results=1)
    messages = [
        SystemMessage(content="You are an AI consultant."),
        HumanMessage(content=f"Context: {retrieved_context}\n\nQuestion: {user_query}")
    ]

    response = chat.invoke(messages)
    memory.save_context({"input": user_query}, {"output": response.content})
    return response.content

# Chat Interface
user_query = st.text_input("Ask a question:")
if st.button("Submit"):
    if user_query:
        response = query_llama3(user_query)
        st.write(f"🤖 Answer: {response}")
    else:
        st.warning("Please enter a question.")

