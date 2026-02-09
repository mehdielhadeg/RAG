# =======================
# app.py - RAG ChatBot Strict avec Pinecone v5+
# =======================

import os
import streamlit as st
from pathlib import Path
import pdfplumber

import easyocr
from PIL import Image
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage

from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import BM25Retriever
# =======================
# CONFIG
# =======================
DOCS_FOLDER = "documents"
os.makedirs(DOCS_FOLDER, exist_ok=True)

PINECONE_API_KEY = ""        # 🔑 remplace par ta clé
PINECONE_INDEX_NAME = "rag"      # nom de ton index
PINECONE_REGION = "us-east-1"          # exemple : us-east-1
PINECONE_DIM = 384                      # dimension embeddings MiniLM
PINECONE_METRIC = "cosine"              # similarité

# =======================
# INITIALISER PINECONE
# =======================
pc = PineconeClient(api_key=PINECONE_API_KEY)

if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION)
    )

# =======================
# MODELS
# =======================
@st.cache_resource
def load_models():
    from huggingface_hub import login
    hf_token = ""
    login(token=hf_token)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.0,
        max_new_tokens=512,
    )

    return embeddings, ChatHuggingFace(llm=llm)


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["fr", "en"], gpu=False)


embeddings, chat_model = load_models()
ocr_reader = load_ocr()

# =======================
# VECTORSTORE (Pinecone)
# =======================
@st.cache_resource
def load_vectorstore():
    # Correct way to target the index with the Pinecone client instance
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, 
        embedding=embeddings, 
        pinecone_api_key=PINECONE_API_KEY
    )
vectorstore = load_vectorstore()
# =======================
# PDF → TEXT
# =======================
def pdf_to_text(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# =======================
# IMAGE → TEXT (OCR)
# =======================
def image_to_text(path):
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)
    results = ocr_reader.readtext(img_np)
    return " ".join([r[1] for r in results])

# =======================
# INGEST
# =======================
def ingest_file(file_path):
    ext = file_path.suffix.lower()

    if ext == ".pdf":
        text = pdf_to_text(file_path)
    elif ext in [".png", ".jpg", ".jpeg"]:
        text = image_to_text(file_path)
    else:
        return "Format non supporté"

    if not text.strip():
        return "Document vide"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = [
        Document(
            page_content=chunk,
            metadata={"source": file_path.name}
        )
        for chunk in splitter.split_text(text)
    ]

    vectorstore.add_documents(docs)
    return f"Chargé : {file_path.name} ({len(docs)} chunks)"

# =======================
# PROMPT STRICT
# =======================
def build_messages(question, context):
    return [
        SystemMessage(
            content=(
                "Tu es un assistant expert en analyse de documents.\n"
                "Reponds UNIQUEMENT en FRANÇAIS.\n"
                "TON OBJECTIF : Répondre à la question de manière concise et structurée en utilisant UNIQUEMENT le contexte fourni.\n"
                "CONSIGNE : Si le contexte contient des informations sur plusieurs sujets différents, ne sélectionne que celles qui répondent directement à la question.\n"
                "Si la réponse n'est pas dans le texte, dis 'Je ne trouve pas cette information'."
            )
        ),
        {
            "role": "user",
            "content": f"CONTEXTE :\n{context}\n\nQUESTION : {question}\n\nRÉPONSE (en français) :"
        }
    ]

# =======================
# UI
# =======================
# =======================
# INITIALISATION STATE (À mettre avant le reste du UI)
# =======================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "all_docs" not in st.session_state:
    st.session_state.all_docs = []


st.set_page_config("RAG ChatBot", "🏥", layout="centered")
st.title("🏥 RAG ChatBot — Strict Mode")

with st.sidebar:
    st.header("📂 Documents")

    uploads = st.file_uploader(
        "Ajouter PDF / Images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    if uploads:
        for f in uploads:
            p = Path(DOCS_FOLDER) / f.name
            p.write_bytes(f.getbuffer())
            # On récupère les chunks créés
            new_chunks = ingest_file(p) 
            # On les ajoute à la liste globale pour BM25
            if new_chunks:
                st.session_state.all_docs.extend(new_chunks)
            st.success(f"Indexé : {f.name}")

    if st.button("🔄 Recharger"):
        for p in Path(DOCS_FOLDER).iterdir():
            ingest_file(p)
        st.rerun()

    if st.button("🧹 Nouvelle conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()


#hybrid retrieval logic: combine vector search and keyword search results
from flashrank import Ranker, RerankRequest

# 1. Initialize a tiny local reranker (runs on CPU, very fast)
@st.cache_resource
def load_reranker():
    return Ranker()

ranker = load_reranker()


def hybrid_retrieval_logic(question, vectorstore, all_docs, k=15):
    """
    Manually combines Vector and Keyword search results.
    """
    # 1. Semantic Search (Pinecone)
    semantic_docs = vectorstore.similarity_search(question, k=k)
    
    # 2. Keyword Search (BM25)
    # BM25Retriever needs a list of Document objects
    if all_docs:
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = k
        keyword_docs = bm25_retriever.invoke(question)
    else:
        keyword_docs = []
        
    # 3. Combine and Deduplicate (based on page_content)
    combined_docs = semantic_docs + keyword_docs
    unique_docs = []
    seen_content = set()
    
    for doc in combined_docs:
        if doc.page_content not in seen_content:
            unique_docs.append(doc)
            seen_content.add(doc.page_content)
            
    return unique_docs

def get_assistant_response(question, vectorstore, all_docs, ranker, chat_model):
    # --- NEW: Query Expansion ---
    expansion_prompt = f"Génère 3 variantes ou mots-clés (en français) pour cette question RAG : '{question}'. Réponds juste les variantes séparées par des virgules."
    variants = chat_model.invoke(expansion_prompt).content.strip()
    search_query = f"{question}, {variants}" # We combine them for a wider search

    # 1. Hybrid Retrieval (Using the expanded query)
    raw_docs = hybrid_retrieval_logic(search_query, vectorstore, all_docs, k=20)
    
    # 2. Format for FlashRank
    passages = [
        {"id": i, "text": d.page_content, "meta": d.metadata} 
        for i, d in enumerate(raw_docs)
    ]
    
    # 3. Rerank (Use the ORIGINAL question for ranking to keep focus)
    rerank_request = RerankRequest(query=question, passages=passages)
    results = ranker.rerank(rerank_request)
    
    top_results = results[:6] # Increased to 6 to give more context
    
    # 4. Context & Sources
    context = "\n\n".join([f"[Source: {r['meta'].get('source')}] {r['text']}" for r in top_results])
    sources = list({r['meta'].get('source') for r in top_results})
    
    # 5. LLM Call
    messages = build_messages(question, context)
    response = chat_model.invoke(messages)
    
    return response.content.strip(), sources




# =======================
# CHAT
# =======================
# --- Display History ---
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            st.caption(f"Sources : {', '.join(m['sources'])}")

# --- User Input ---
if question := st.chat_input("Votre question"):
    # Save User Message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Process Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Recherche hybride & Analyse..."):
            # Call the separated logic
            answer, sources = get_assistant_response(
                question, 
                vectorstore, 
                st.session_state.get("all_docs", []), 
                ranker, 
                chat_model
            )
            
            # Display
            st.markdown(answer)
            if sources:
                st.caption(f"Sources : {', '.join(sources)}")
    
    # Save Assistant Message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })
