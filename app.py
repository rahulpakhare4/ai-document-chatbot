import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings  # ✅ FIXED IMPORT!
from PyPDF2 import PdfReader
import chromadb
from chromadb.config import Settings

# Use in-memory database
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # ✅ Fix for Streamlit


st.title("📄 AI Chatbot with PDF Support")

# Dummy Collection
collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

# Upload PDF
uploaded_file = st.file_uploader("📂 Upload a PDF", type="pdf")

if uploaded_file:
    # Extract text from PDF
    reader = PdfReader(uploaded_file)
    extracted_text = "\n".join([page.extract_text() or "" for page in reader.pages])
    st.success(f"✅ Extracted text from {len(reader.pages)} pages!")

    # Use ChromaDB with in-memory storage
    chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))  # ✅ FIXED!
    collection = chroma_client.get_or_create_collection(name="ai_knowledge_base")

    # Load HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # ✅ SPECIFY MODEL

    # Generate embeddings and store in ChromaDB
    doc_embedding = embeddings.embed_documents([extracted_text])
    collection.add(
        ids=["pdf_doc_1"],
        embeddings=doc_embedding,
        metadatas=[{"source": "uploaded_pdf"}]
    )

    st.success("✅ Document indexed successfully!")

# User Query
user_query = st.text_input("💬 Ask a question:")
if st.button("🔍 Get Answer"):
    if not uploaded_file:
        st.error("⚠️ Please upload a PDF first!")
    else:
        # Retrieve relevant chunk
        results = collection.query(query_texts=[user_query], n_results=1)
        best_match = results["documents"][0][0] if results["documents"] else "No relevant information found."
        
        st.write("🤖 **Answer:**", best_match)
