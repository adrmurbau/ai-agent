# backend/agent.py
import os
import re
from typing import List, Tuple
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

try:
    import pypdf
    HAS_PYPDF = True
except Exception:
    HAS_PYPDF = False

CHUNK_SIZE = 800      # caracteres
CHUNK_OVERLAP = 150   # solape entre chunks

def read_text_from_file(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in [".pdf"] and HAS_PYPDF:
        reader = pypdf.PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""  # ignora otros formatos en este MVP

def clean_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def chunk_text(t: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(t):
        end = min(start + size, len(t))
        chunks.append(t[start:end])
        if end == len(t):
            break
        start = end - overlap
    return chunks

class RAGIndex:
    def __init__(self):
        self.docs: List[str] = []
        self.vectorizer = None
        self.doc_matrix = None

    def build(self, paths: List[str]):
        raw_texts = []
        for p in paths:
            txt = read_text_from_file(Path(p))
            if not txt:
                continue
            raw_texts.append(clean_text(txt))
        # trocear documentos
        chunks = []
        for t in raw_texts:
            chunks.extend(chunk_text(t))
        self.docs = chunks if chunks else ["No documents provided."]
        self.vectorizer = TfidfVectorizer(stop_words=None, max_features=20000)
        self.doc_matrix = self.vectorizer.fit_transform(self.docs)

    def top_k(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if self.vectorizer is None or self.doc_matrix is None:
            return [("Index is empty. Please ingest documents.", 0.0)]
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = np.argsort(sims)[::-1][:k]
        return [(self.docs[i], float(sims[i])) for i in idx]

class Generator:
    def __init__(self, model_name: str = "google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        out = self.pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()

SYSTEM_PROMPT = (
    "Eres un asistente útil. Debes responder SIEMPRE en español. "
    "Usa ÚNICAMENTE el contexto proporcionado para responder. "
    "Si la respuesta no está claramente en el contexto, responde literalmente: "
    "'No lo sé con el contexto proporcionado'."
)

class Agent:
    def __init__(self):
        self.index = RAGIndex()
        self.generator = Generator()

    def ingest(self, file_paths: List[str]) -> int:
        self.index.build(file_paths)
        return len(self.index.docs)

    def answer(self, question: str, k: int = 5) -> str:
        top = self.index.top_k(question, k=k)
        context = "\n\n".join([c for c, _ in top])
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {question}\n\n"
            f"Respuesta (en español):"
        )
        return self.generator.generate(prompt, max_new_tokens=256)
