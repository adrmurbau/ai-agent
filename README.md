# 🧠 Free AI Agent – RAG + FLAN-T5 (100% Free, No API Keys)

Agente de IA que responde preguntas sobre tus documentos **PDF / TXT** usando un pipeline completamente gratuito:

- 🔎 **RAG ligero** con recuperación por **TF-IDF** (scikit-learn)  
- 🤖 **FLAN-T5** como modelo generador (Transformers, CPU)  
- ⚙️ **Backend en FastAPI** completamente dockerizado  
- 🖥️ **Interfaz gráfica en Gradio**  
- 🧪 Funciona **offline**, sin tokens ni APIs externas  

Este proyecto está pensado como un **MVP limpio y entendible** para portfolio: enseña cómo construir un agente RAG desde cero usando solo herramientas gratuitas.

---

## ✨ Funcionalidades principales

- Ingesta de uno o varios documentos (TXT / PDF con texto)
- Chunking automático del contenido
- Creación de índice TF-IDF para recuperar contexto relevante
- Respuestas generadas usando solo la información del documento
- API REST (`/ingest` y `/ask`)
- UI visual en Gradio para uso cómodo
- Docker para despliegue local reproducible

---

## 📦 Backend (FastAPI + Docker)

Desde la **raíz** del proyecto:

```bash
docker build -t ai-agent-backend ./backend
docker run --rm -p 8000:8000 ai-agent-backend
```

El backend estará disponible en:

👉 http://localhost:8000/docs

Desde ahí puedes probar los endpoints /ingest y /ask.

## 🖥️ UI opcional (Gradio)
En otra terminal:

```bash
py -m pip install -r backend/requirements.txt
py app.py
```

La interfaz estará disponible en:

👉 http://localhost:7860

## 🧪 Ejemplo de uso con cURL
1) Documento de prueba
```bash
echo "El aprendizaje supervisado usa pares entrada-salida." > sample_docs/doc1.txt
```
2) Ingesta:
```bash
curl -X POST "http://localhost:8000/ingest" -F "files=@sample_docs/doc1.txt"
```
3) Pregunta:
```bash
curl -X POST "http://localhost:8000/ask" -F "question=¿De qué trata el documento?" -F "k=3"
```
Respuesta típica:
```json
{"answer": "El documento trata sobre el aprendizaje supervisado y el uso de pares entrada-salida."}
```
---
## 📂 Estructura del proyecto
```bash
/
├── backend/
│   ├── agent.py           # Lógica del RAG + LLM
│   ├── main.py            # FastAPI
│   ├── requirements.txt
│   └── Dockerfile
├── app.py                 # UI con Gradio
├── sample_docs/           # Documentos de ejemplo
├── screenshots/            # (Opcional) Capturas para el README
└── README.md
```
---
## 🧱 Arquitectura 
1. Ingesta
Lee PDFs/TXTs → limpia texto → genera chunks → construye TF-IDF.

2. Consulta
* Recupera los 𝑘 chunks más similares mediante coseno.
* Construye un prompt con el contexto recuperado.
* Genera respuesta usando FLAN-T5 en CPU.

3.Entrega
* API REST vía FastAPI
* UI opcional vía Gradio
---
## 🔧 Tecnologías usadas
* **Python 3.11**
* **FastAPI**
* **Transformers (Hugging Face)**
* **scikit-learn**
* **pypdf**
* **Gradio**
* **Docker**
---
## 🚀 Mejoras futuras (roadmap)
* Añadir stopwords en español al TF-IDF
* Resaltar el pasaje usado en la respuesta
* Soporte para CSV/JSON
* Guardar el índice para no re-ingestar
* Demo gratuita en Hugging Face Spaces
---
## 📜 Licencia
MIT License – libre para usar, modificar y ampliar.

## 👤 Autor
Adrián Muriel Bautista
- GitHub: https://github.com/adrmurbau
- LinkedIn: https://www.linkedin.com/in/adrmurbau
