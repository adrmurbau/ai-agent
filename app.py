# app.py
import gradio as gr
import requests

API_URL = "http://localhost:8000"  # en Docker Compose podrías cambiarlo al servicio backend

def ingest_files(files):
    if not files:
        return "Selecciona 1-2 documentos."
    files_to_send = [("files", (f.name, open(f.name, "rb"), "application/octet-stream")) for f in files]
    r = requests.post(f"{API_URL}/ingest", files=files_to_send, timeout=300)
    if r.ok:
        data = r.json()
        return f"Índice creado con {data['chunks']} chunks a partir de: {', '.join(data['files'])}"
    return f"Error: {r.text}"

def ask_question(q, k):
    r = requests.post(f"{API_URL}/ask", data={"question": q, "k": int(k)}, timeout=300)
    if r.ok:
        return r.json()["answer"]
    return f"Error: {r.text}"

with gr.Blocks(title="Free AI Agent (RAG + FLAN-T5)") as demo:
    gr.Markdown("# Free AI Agent (RAG + FLAN-T5)\nSube PDFs/TXT y pregunta sobre su contenido. 100% gratis, sin API keys.")
    with gr.Row():
        files = gr.File(file_count="multiple", type="filepath", label="Documentos (.pdf, .txt)")
        ingest_btn = gr.Button("Crear índice")
    ingest_out = gr.Textbox(label="Estado", lines=2)

    with gr.Row():
        q = gr.Textbox(label="Pregunta")
        k = gr.Slider(1, 8, value=5, step=1, label="Top-K contexto")
    ask_btn = gr.Button("Preguntar")
    ans = gr.Textbox(label="Respuesta", lines=10)

    ingest_btn.click(ingest_files, inputs=[files], outputs=[ingest_out])
    ask_btn.click(ask_question, inputs=[q, k], outputs=[ans])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
