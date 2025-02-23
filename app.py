import streamlit as st
import chromadb
import os
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util
from PyPDF2 import PdfReader

# Streamlit UI
st.set_page_config(page_title="AI Agent", layout="wide")
st.title("📄 AI-Powered Document Chatbot")

# ✅ Initialize ChromaDB
@st.cache_resource
def initialize_chromadb():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="ai_knowledge_base")

collection = initialize_chromadb()

# ✅ Initialize HuggingFace Embedding Model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ✅ Initialize Sentence-Transformers Model for Semantic Matching
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# ✅ Initialize Groq Chat Model
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="your_groq_api_key")

# ✅ Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ✅ Function to Extract Text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        return f"⚠️ Error reading PDF: {str(e)}"

# ✅ Function to Chunk Text
def chunk_text(text, chunk_size=600):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    return splitter.split_text(text)

# ✅ Store Embeddings in ChromaDB
def store_embeddings(chunks, collection, embedding_model):
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]

    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )
        return f"✅ Stored {len(new_chunks)} new embeddings!"
    return "⚠️ No new chunks to add."

# ✅ Retrieve Context from ChromaDB
def retrieve_context(query, top_k=1):
    query_embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

# ✅ Evaluate Response Using Semantic Similarity
def evaluate_response(user_query, generated_response, context):
    response_embedding = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_embedding = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_embedding, context_embedding)[0][0].item()

# ✅ Query AI Model
def query_ai(user_query):
    system_prompt = """
    System Prompt: You are a AI clone of Rahul Pakhare, a Consultant with 11+ years of experience.
    Respond in a natural, engaging tone.
    """

    past_chat_history = memory.load_memory_variables({}).get("chat_history", [])[-8:]
    retrieved_context = retrieve_context(user_query)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"🗂 Past Chat: {past_chat_history}\n📖 Context: {retrieved_context}\n📝 Question: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        evaluation_score = evaluate_response(user_query, response.content, retrieved_context)

        return response.content, evaluation_score
    except Exception as e:
        return f"⚠️ API Error: {str(e)}", 0

# ✅ Upload PDF in Streamlit
uploaded_file = st.file_uploader("📂 Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_text = extract_text_from_pdf(uploaded_file)
    st.success(f"✅ Extracted {len(pdf_text.split())} words from PDF!")
    
    chunks = chunk_text(pdf_text)
    embedding_status = store_embeddings(chunks, collection, embedding_model)
    st.info(embedding_status)

# ✅ Chat Interface
st.header("💬 Chat with AI")
user_query = st.text_input("📝 Ask a question:")
if st.button("🔍 Get Answer") and user_query:
    response, score = query_ai(user_query)
    st.markdown(f"**🤖 AI:** {response}")
    st.markdown(f"📊 **Confidence Score:** {score:.2f}")

# ✅ Display Chat History
if st.sidebar.checkbox("Show Chat History"):
    past_chat = memory.load_memory_variables({}).get("chat_history", [])
    st.sidebar.write("🗂 **Chat History:**", past_chat)

