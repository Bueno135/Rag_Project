from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from app.services.cache import create_cache, get_cache, set_cache
from app.services.memory import add_message, get_history, clear_history
import numpy as np
import os
import sys

# garante import relativo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

app = FastAPI(title="RAG PDF Assistant")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

create_cache()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile):
    file_path = os.path.join(DATA_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"message": f"Arquivo salvo como {file.filename}"}

@app.post("/clear_session")
async def clear_session(session_id: str = Form(...)):
    clear_history(session_id)
    return {"message": f"Histórico da sessão {session_id} apagado"}

@app.post("/ask")
async def ask(
    question: str = Form(...),
    filename: str = Form(...),
    session_id: str = Form(...),
):
    # 1. histórico recente
    history = get_history(session_id)
    history_text = "\n".join(
        [f"{m['role'].upper()}: {m['text']}" for m in history]
    )

    # 2. prompt com contexto
    contextualized_question = (
        "Contexto da conversa até agora:\n"
        f"{history_text}\n\n"
        "Nova pergunta do usuário:\n"
        f"{question}\n"
        "Responda de forma útil, direta e baseada no documento."
    )

    # 3. cache check
    cached = get_cache(contextualized_question)
    if cached:
        answer_text = cached["text"]
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", answer_text)
        return {"source": "cache", "answer": answer_text}

    # 4. carrega PDF
    pdf_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(pdf_path):
        return {"error": f"Arquivo {filename} não encontrado."}

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]

    # 5. embeddings + similarity search
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents(texts)
    query_vector = embeddings.embed_query(contextualized_question)

    cosine_scores = cosine_similarity([query_vector], vectors)[0]
    best_idx = int(np.argmax(cosine_scores))
    best_text = texts[best_idx][:700]

    # 6. salva no histórico
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", best_text)

    # 7. salva no cache
    set_cache(contextualized_question, {"text": best_text})

    return {"source": "model", "answer": best_text, "session_id": session_id}
