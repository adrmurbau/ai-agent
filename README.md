# 🧠 Free AI Agent – RAG + FLAN-T5 (No API keys)

Agente de IA que responde preguntas sobre tus documentos PDF/TXT usando:
- Recuperación de contexto con **TF-IDF** (scikit-learn)
- Generación de respuestas con **google/flan-t5-base** (Transformers, CPU)
- Backend en **FastAPI** (Docker)
- Interfaz opcional en **Gradio**
- 🚫 **Sin API keys** – 100% gratuito

## ✨ Funcionalidades
- Ingesta de documentos (TXT / PDF con texto).
- Construcción automática de índices con chunks.
- Búsqueda de contexto usando TF-IDF (top-k).
- Generación de respuestas basadas SOLO en el contexto.
- API REST con `/ingest` y `/ask`.
- UI visual en Gradio.

## 📦 Ejecutar backend (Docker)

Desde la raíz del proyecto:

```bash
docker build -t ai-agent-backend ./backend
docker run --rm -p 8000:8000 ai-agent-backend

## 🚀 Ejecutar la UI (Gradio)

Con el backend ya corriendo en `http://localhost:8000`:

```bash
# Instalar dependencias en tu máquina
py -m pip install -r backend/requirements.txt

# Lanzar la UI
python app.py
