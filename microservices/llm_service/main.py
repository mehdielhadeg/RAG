from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from huggingface_hub import login

app = FastAPI(title="Medical RAG - LLM Service")

HF_TOKEN = "" 
login(token=HF_TOKEN)

# Initialize Model
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.1,
    max_new_tokens=1024,
)
chat_model = ChatHuggingFace(llm=llm)

class ChatRequest(BaseModel):
    prompt: str
    context: list[str]

@app.post("/chat")
async def chat(data: ChatRequest):
    try:
        context_str = "\n\n".join(data.context)
        system_msg = "Tu es un assistant. Réponds en français à partir du contexte fourni. si la reponse n'existe pas dis qu'elle n'existe pas"
        user_msg = f"CONTEXTE:\n{context_str}\n\nQUESTION: {data.prompt}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        response = chat_model.invoke(messages)
        return {"answer": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
