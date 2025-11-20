# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from agent import Agent
import tempfile
import shutil
from pathlib import Path

app = FastAPI(title="Free AI Agent (RAG + FLAN-T5)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

agent = Agent()

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    tmpdir = Path(tempfile.mkdtemp())
    saved = []
    for uf in files:
        dest = tmpdir / uf.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        saved.append(str(dest))
    num_chunks = agent.ingest(saved)
    return {"status": "ok", "chunks": num_chunks, "files": [Path(p).name for p in saved]}

@app.post("/ask")
async def ask(question: str = Form(...), k: int = Form(5)):
    answer = agent.answer(question, k=k)
    return {"answer": answer}
