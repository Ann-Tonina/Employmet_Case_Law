__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Kenya Law Discovery", page_icon="⚖️", layout="wide")

# --- 2. AUTHENTICATION (The fix that worked) ---
def get_verified_key():
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key: return env_key
    if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
    return None

api_key = get_verified_key()
if not api_key:
    st.error("❌ API Key Missing. Please set the Windows Environment Variable.")
    st.stop()

# --- 3. RESOURCE CACHING ---
@st.cache_resource
def load_system():
    # Configure Gemini 3 Flash (2026 flagship model)
    genai.configure(api_key=api_key)
    llm = genai.GenerativeModel('gemini-3-flash-preview')
    
    # Load Sentence Transformer for legal embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Initialize Persistent ChromaDB
    # Update path to "./kenya_law_db" if that is your folder name
    client = chromadb.PersistentClient(path="./database")
    collection = client.get_or_create_collection(name="employment_case_law")
    
    return llm, embedder, collection

llm, embedder, collection = load_system()

# --- 4. USER INTERFACE ---
st.title("⚖️ Kenyan Employment Law Discovery")
st.markdown("Search ELRC judgments and get AI-summarized legal precedents.")

query = st.text_input("Ask a question (e.g., 'What are the grounds for summary dismissal?'):")

if query:
    with st.spinner("Searching case law and reasoning..."):
        try:
            # Step A: Semantic Vector Search
            # We encode the query to match the math of the database
            query_vec = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=query_vec, n_results=3)
            
            # Step B: Build Legal Context
            context = ""
            if results['documents']:
                for doc in results['documents'][0]:
                    context += f"\n[DOCUMENT START]\n{doc}\n[DOCUMENT END]\n"
            
            # Step C: Gemini 3 Reasoning
            # In 2026, we prompt the model to act as a Kenyan High Court Researcher
            prompt = f"""
            You are a Senior Researcher for the Kenyan High Court. 
            Answer the user's question based ONLY on the provided ELRC judgments. 
            Cite the case names if available in the text.

            CONTEXT FROM DATABASE:
            {context}

            QUESTION:
            {query}
            """
            
            # Use 'minimal' thinking for high speed, or 'medium' for deep legal analysis
            response = llm.generate_content(prompt)
            
            # Step D: Display Results
            st.subheader("Legal Analysis")
            st.markdown(response.text)
            
            with st.expander("📚 View Original Case Excerpts"):
                for doc in results['documents'][0]:
                    st.info(doc)
                    
        except Exception as e:
            st.error(f"Error during search: {e}")

# --- FOOTER ---
st.sidebar.caption("System Status: Online (Gemini 3 Flash)")