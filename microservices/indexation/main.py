import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# LangChain Imports
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = FastAPI(title="Medical RAG - Indexing Service (Hybrid)")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
STORE_PATH = "vector_store"

# Data Models
class IndexReq(BaseModel):
    text: str
    filename: str

class SearchReq(BaseModel):
    query: str
    top_k: int = 4

def rrf_fusion(results_vector: List[Document], results_bm25: List[Document], k: int = 60):
    """
    Reciprocal Rank Fusion (RRF) algorithm to combine scores 
    from Vector and Keyword searches manually.
    """
    scores = {}
    
    # Process Vector Results
    for rank, doc in enumerate(results_vector, 1):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
        
    # Process BM25 Results
    for rank, doc in enumerate(results_bm25, 1):
        content = doc.page_content
        scores[content] = scores.get(content, 0) + 1 / (k + rank)
    
    # Sort documents by fused RRF score
    sorted_content = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [content for content, score in sorted_content]

@app.post("/index")
async def index_document(data: IndexReq):
    try:
        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(data.text)
        docs = [Document(page_content=c, metadata={"source": data.filename}) for c in chunks]

        # Update or Create FAISS Index
        if os.path.exists(os.path.join(STORE_PATH, "index.faiss")):
            db = FAISS.load_local(STORE_PATH, embeddings, allow_dangerous_deserialization=True)
            db.add_documents(docs)
        else:
            db = FAISS.from_documents(docs, embeddings)
        
        db.save_local(STORE_PATH)
        return {"status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def hybrid_search(data: SearchReq):
    if not os.path.exists(os.path.join(STORE_PATH, "index.faiss")):
        return {"context": [], "message": "Index empty"}

    try:
        # 1. Semantic Search (FAISS)
        db = FAISS.load_local(STORE_PATH, embeddings, allow_dangerous_deserialization=True)
        vector_results = db.similarity_search(data.query, k=10)

        # 2. Keyword Search (BM25)
        # We rebuild the BM25 index from current documents in FAISS for accuracy
        all_docs = list(db.docstore._dict.values())
        bm25 = BM25Retriever.from_documents(all_docs)
        bm25.k = 10
        bm25_results = bm25.invoke(data.query)

        # 3. Apply Fusion Logic
        fused_context = rrf_fusion(vector_results, bm25_results)
        
        return {"context": fused_context[:data.top_k]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- THE ENTRY POINT ---
if __name__ == "__main__":
    # This ensures the service runs on port 8002
    print("Starting Indexing Service on http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)
