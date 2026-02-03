import streamlit as st
import requests
from pathlib import Path

# URLs of our Microservices
INGESTION_URL = "http://localhost:8001/extract"
INDEXING_URL = "http://localhost:8002"
LLM_URL = "http://localhost:8003/chat"

st.set_page_config(page_title=" RAG Microservices", page_icon="🏥")
st.title("🏥 Assistant Distribué")

# --- SIDEBAR: DOCUMENT MANAGEMENT ---
with st.sidebar:
    st.header("📂 Gestion des Documents")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "jpg", "png", "jpeg"], accept_multiple_files=True)
    
    if st.button("Lancer l'Analyse"):
        if uploaded_files:
            for f in uploaded_files:
                with st.spinner(f"Traitement de {f.name}..."):
                    # 1. Ingestion (OCR/PDF Parsing)
                    files = {"file": (f.name, f.getvalue(), f.type)}
                    ingest_res = requests.post(INGESTION_URL, files=files)
                    
                    if ingest_res.status_code == 200:
                        extracted_text = ingest_res.json()["text"]
                        
                        # 2. Indexing (Vector Store)
                        index_payload = {"text": extracted_text, "filename": f.name}
                        idx_res = requests.post(f"{INDEXING_URL}/index", json=index_payload)
                        
                        if idx_res.status_code == 200:
                            st.success(f"✅ {f.name} indexé")
                        else:
                            st.error(f"❌ Erreur Indexation: {idx_res.text}")
                    else:
                        st.error(f"❌ Erreur Extraction: {ingest_res.text}")
        else:
            st.warning("Ajoutez des fichiers.")

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            # 3. Search (Retrieve context from Indexer)
            search_res = requests.post(f"{INDEXING_URL}/search", json={"query": prompt, "k": 4})
            
            if search_res.status_code == 200:
                context = search_res.json()["context"]
                
                # 4. LLM Generation
                llm_payload = {"prompt": prompt, "context": context}
                llm_res = requests.post(LLM_URL, json=llm_payload)
                
                if llm_res.status_code == 200:
                    answer = llm_res.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Erreur LLM")
            else:
                st.error("L'index est probablement vide. Chargez des documents.")