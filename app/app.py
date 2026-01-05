import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Kenya Employment Law Bot", page_icon="⚖️", layout="wide")

# --- LOAD MODELS & DB (Cached to stay fast) ---
@st.cache_resource
def load_resources():
    # Load Embedding Model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Connect to ChromaDB
    db_path = "./kenya_law_db"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name="employment_case_law")
    
    # Configure Gemini
    # Note: In production, use st.secrets for the API key
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
    llm = genai.GenerativeModel('gemini-3-flash-preview')
    
    return model, collection, llm

model, collection, llm = load_resources()

# --- UI HEADER ---
st.title("⚖️ Kenya Employment Law Research Bot")
st.markdown("Find authoritative answers with citations from the **ELRC** and **Court of Appeal**.")

# --- SIDEBAR (History & Info) ---
with st.sidebar:
    st.header("Search Settings")
    n_results = st.slider("Number of cases to cite", 1, 5, 3)
    st.info("This tool uses Retrieval-Augmented Generation (RAG) to ensure answers are based on actual Kenyan case law.")

# --- SEARCH INTERFACE ---
query = st.text_input("Describe the legal issue (e.g., 'Procedure for unfair dismissal'):")

if query:
    with st.spinner("Searching case law and generating answer..."):
        # 1. Retrieval
        query_embedding = model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        
        # 2. Build Context
        context_parts = []
        for i in range(len(results['documents'][0])):
            doc = results['documents'][0][i]
            meta = results['metadatas'][0][i]
            context_parts.append(f"Source [{meta['case_title']}]:\n{doc}")
        
        full_context = "\n\n".join(context_parts)
        
        # 3. LLM Generation
        prompt = f"Use the following Kenyan legal context to answer: {query}\n\nCONTEXT:\n{full_context}"
        response = llm.generate_content(prompt)
        
        # 4. Display Results
        st.subheader("Legal Findings")
        st.write(response.text)
        
        # 5. Citations Expander
        with st.expander("View Cited Sources"):
            for i in range(len(results['metadatas'][0])):
                m = results['metadatas'][0][i]
                st.markdown(f"**{m['case_title']}**")
                st.caption(f"Court: {m['court']} | Date: {m['decision_date']}")
                st.write(results['documents'][0][i][:300] + "...")
                st.divider()