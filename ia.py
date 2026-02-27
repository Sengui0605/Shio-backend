from fastapi import FastAPI, Request, File, UploadFile, Body, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import asyncio
import os
import json
import logging
import subprocess
import random
import shlex
from datetime import datetime
import re
import uuid
import time
import math
from urllib.parse import quote_plus
from collections import defaultdict

import ollama
from faster_whisper import WhisperModel
from bs4 import BeautifulSoup
import requests
from duckduckgo_search import DDGS
import PyPDF2

try:
    from aprendizaje import shio_programar
except Exception as e:
    print("Error importando aprendizaje.py:", e)
    shio_programar = None

# ── LOGGING WITH BUFFER ──
class LogBuffer(logging.Handler):
    def __init__(self, max_lines=200):
        super().__init__()
        self.logs = []
        self.max_lines = max_lines
        self.clients: list[WebSocket] = []

    def emit(self, record):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": record.levelname,
            "msg": self.format(record)
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_lines:
            self.logs = self.logs[-self.max_lines:]
        # Notify WebSocket clients
        for ws in list(self.clients):
            try:
                asyncio.get_event_loop().create_task(ws.send_json(entry))
            except:
                pass

log_buffer = LogBuffer()
log_buffer.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(), log_buffer])

app = FastAPI(title="Shio AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CONFIG ──
CONFIG_FILE = "shio_config.json"

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    default = {"pin": "1234"}
    save_config(default)
    return default

def save_config(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

CONFIG = load_config()

# ── PROMPT ──
PROMPT = ""
try:
    with open("prompt.txt", "r", encoding="utf-8") as f:
        PROMPT = f.read()
except: pass

IDENTIDAD = (
    "Eres SHIO, una IA local avanzada creada por Senjiro.\n"
    "Nunca seas vaga. Nunca generes código inútil.\n"
    "Si programas, crea archivos reales organizados."
)

try:
    APPS = json.load(open("apps_config.json", "r", encoding="utf-8"))
except:
    APPS = []

BANNED = ["del", "format", "shutdown", "rd ", "rmdir", "diskpart", "bcdedit", "reg delete", "mkfs"]

# ── RATE LIMITING ──
rate_limits: dict[str, list[float]] = defaultdict(list)
RATE_MAX = 30
RATE_WINDOW = 60

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    rate_limits[ip] = [t for t in rate_limits[ip] if now - t < RATE_WINDOW]
    if len(rate_limits[ip]) >= RATE_MAX:
        return False
    rate_limits[ip].append(now)
    return True

# ── AUTH MIDDLEWARE ──
AUTH_EXCLUDED = ["/auth", "/health", "/docs", "/openapi.json"]

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(e) for e in AUTH_EXCLUDED):
        return await call_next(request)
    if request.method == "OPTIONS":
        return await call_next(request)

    pin = request.headers.get("Authorization", "").replace("Bearer ", "")
    if pin != CONFIG.get("pin", "1234"):
        return JSONResponse(status_code=401, content={"error": "No autorizado"})

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"error": "Demasiadas peticiones. Espera un momento."})

    return await call_next(request)

# ── PERSISTENT HISTORY ──
HISTORY_FILE = "chat_history.json"

def load_all_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

def save_all_history(data: list):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_conversation(session_id: str) -> dict | None:
    history = load_all_history()
    for conv in history:
        if conv["id"] == session_id:
            return conv
    return None

def save_conversation(conv: dict):
    history = load_all_history()
    found = False
    for i, c in enumerate(history):
        if c["id"] == conv["id"]:
            history[i] = conv
            found = True
            break
    if not found:
        history.append(conv)
    save_all_history(history)

def generate_title(msg: str) -> str:
    title = msg.strip()[:50]
    if len(msg.strip()) > 50:
        title += "..."
    return title

HISTORIA = {}
stt_model = WhisperModel("tiny", device="cpu", compute_type="int8")

# ── UPLOADS ──
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
    except Exception as e:
        text = f"Error leyendo PDF: {e}"
    return text.strip()

def extract_text_from_file(path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    text_exts = [".py", ".js", ".ts", ".html", ".css", ".json", ".md", ".txt", ".csv", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".bat", ".sh", ".sql", ".java", ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".php"]
    if ext in text_exts:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except:
            return "Error leyendo archivo"
    return f"Tipo de archivo no soportado: {ext}"

# ── RAG SYSTEM ──
RAG_INDEX_FILE = "rag_index.json"

def load_rag_index():
    if os.path.exists(RAG_INDEX_FILE):
        try:
            with open(RAG_INDEX_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"chunks": [], "embeddings": []}
    return {"chunks": [], "embeddings": []}

def save_rag_index(data):
    with open(RAG_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def chunk_text(text: str, chunk_size: int = 500) -> list[str]:
    words = text.split()
    chunks = []
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append(" ".join(current))
    return chunks

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

def rag_search(query: str, top_k: int = 3) -> list[str]:
    index = load_rag_index()
    if not index["chunks"]:
        return []
    try:
        resp = ollama.embed(model="nomic-embed-text", input=query)
        q_emb = resp["embeddings"][0]
    except:
        return []

    scored = []
    for i, emb in enumerate(index["embeddings"]):
        sim = cosine_similarity(q_emb, emb)
        scored.append((sim, index["chunks"][i]))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:top_k]]

# ── APP & CMD ──
def ejecutar_app(nombre: str) -> str:
    for a in APPS:
        if a["name"].lower() == nombre.lower():
            cmd = a["cmd"]
            if comando_peligroso(cmd):
                return "App bloqueada por seguridad."
            try:
                os.startfile(cmd)
                return f"Abriendo {nombre}..."
            except Exception as e:
                return f"Error al abrir {nombre}: {e}"
    return "App no encontrada."

def comando_peligroso(cmd: str) -> bool:
    return any(b in cmd.lower() for b in BANNED)

def ejecutar_cmd(cmd: str) -> str:
    if comando_peligroso(cmd):
        return "Comando bloqueado."
    try:
        args = shlex.split(cmd)
        proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=10)
        return out or err
    except Exception as e:
        return str(e)

# ── WEB SEARCH ──
def buscar_web(query: str, num_results: int = 5) -> list:
    try:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(query, region="es-es", max_results=num_results):
                results.append({
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", "")
                })
            return results
    except Exception as e:
        logging.exception("Error en búsqueda web")
        return []

def format_search_results(results: list, query: str) -> str:
    if not results:
        return f"No encontré resultados para: **{query}**"
    md = f"🔍 **Resultados para:** *{query}*\n\n"
    for i, r in enumerate(results, 1):
        md += f"**{i}. [{r['title']}]({r['url']})**\n"
        if r['snippet']:
            md += f"   {r['snippet']}\n"
        md += "\n"
    return md

def tool_selector(msg: str) -> str:
    msg_lc = msg.lower()
    if msg_lc.startswith("abre "): return "app"
    if msg_lc.startswith("cmd "): return "cmd"
    if msg_lc.startswith("programa "): return "coder"
    if msg_lc.startswith("busca "): return "search"
    return "llm"

def fecha() -> str:
    return datetime.now().strftime("%d/%m/%Y %H:%M")

async def lanzar_auto_programador(tarea: str) -> str:
    if not shio_programar:
        return "Coder no cargado."
    tipo = "python"
    t_lower = tarea.lower()
    if "html" in t_lower: tipo = "html"
    elif "css" in t_lower: tipo = "css"
    elif "javascript" in t_lower or "js" in t_lower: tipo = "js"
    EXTENSIONES = {"python": "py","html": "html","css": "css","js": "js","javascript": "js"}
    carpeta = os.path.join("shio_code", tipo)
    os.makedirs(carpeta, exist_ok=True)
    extension = EXTENSIONES.get(tipo, "txt")
    nombre_archivo = f"{uuid.uuid4().hex}.{extension}"
    ruta_archivo = os.path.join(carpeta, nombre_archivo)
    prompt_real = f"""
TAREA REAL:
{tarea}

REGLAS:
- Crear proyecto real
- Sin comentarios
- Solo devuelve código
- NO menciones rutas ni nombres de archivo
"""
    asyncio.create_task(asyncio.to_thread(shio_programar, prompt_real, ruta_archivo, tipo))
    return f"Coder ejecutándose... Se generará: {nombre_archivo}"

async def chat(msg: str, session_id: str = None, file_context: str = None):
    tool = tool_selector(msg)
    if tool == "app":
        return ejecutar_app(msg.replace("abre ", "")), []
    if tool == "cmd":
        return ejecutar_cmd(msg[4:]), []
    if tool == "coder":
        tarea = msg.replace("programa ", "")
        return await lanzar_auto_programador(tarea), []
    if tool == "search":
        query = msg[6:].strip()
        results = await asyncio.to_thread(buscar_web, query)
        return format_search_results(results, query), []

    if not session_id:
        session_id = str(uuid.uuid4())

    if session_id not in HISTORIA:
        conv = get_conversation(session_id)
        if conv:
            HISTORIA[session_id] = conv["messages"]
        else:
            HISTORIA[session_id] = [
                {"role": "system", "content": IDENTIDAD},
                {"role": "system", "content": "Fecha: " + fecha()},
                {"role": "system", "content": PROMPT},
            ]

    # Build enriched user message with file/RAG context
    enriched_msg = ""

    # File context — prepend to user message so model sees it
    if file_context:
        enriched_msg += f"[ARCHIVO ADJUNTO]\n{file_context[:4000]}\n[FIN DEL ARCHIVO]\n\n"

    # RAG: inject relevant context
    rag_chunks = rag_search(msg)
    if rag_chunks:
        context = "\n---\n".join(rag_chunks)
        enriched_msg += f"[CONTEXTO DE DOCUMENTOS LOCALES]\n{context}\n[FIN DEL CONTEXTO]\n\n"

    enriched_msg += msg

    HISTORIA[session_id].append({"role": "user", "content": enriched_msg})

    try:
        r = await asyncio.to_thread(
            ollama.chat,
            model="gpt-oss:120b-cloud",
            messages=HISTORIA[session_id],
            options={"temperature": 0.2, "num_predict": 1000}
        )
        contenido = "\n".join(line.strip() for line in r["message"]["content"].splitlines() if line.strip())
        HISTORIA[session_id].append({"role": "assistant", "content": contenido})

        conv = get_conversation(session_id)
        first_user_msg = next((m["content"] for m in HISTORIA[session_id] if m["role"] == "user"), msg)
        if conv:
            conv["messages"] = HISTORIA[session_id]
        else:
            conv = {
                "id": session_id,
                "title": generate_title(first_user_msg),
                "created": datetime.now().isoformat(),
                "messages": HISTORIA[session_id]
            }
        save_conversation(conv)

        return contenido, [], session_id
    except Exception as e:
        logging.exception("Error IA")
        return "Error IA", [], session_id

# ══════════════════════════════════════════
#                ENDPOINTS
# ══════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/auth")
async def auth(request: Request):
    data = await request.json()
    pin = data.get("pin", "")
    if pin == CONFIG.get("pin", "1234"):
        return {"ok": True, "pin": pin}
    return JSONResponse(status_code=401, content={"error": "PIN incorrecto"})

@app.post("/chat")
async def chat_post(request: Request):
    data = await request.json()
    msg = data.get("msg", "")
    session_id = data.get("session_id")
    file_context = data.get("file_context")
    result = await chat(msg, session_id=session_id, file_context=file_context)
    if len(result) == 3:
        text, archivos, sid = result
    else:
        text, archivos = result
        sid = session_id
    resp = {"text": text, "session_id": sid}
    if archivos: resp["file"] = archivos[0]
    return resp

@app.post("/stt")
async def stt(file: UploadFile = File(...)):
    audio = await file.read()
    if len(audio) > 10_000_000:
        return {"error": "Archivo demasiado grande"}
    tmp_file = f"voz_{uuid.uuid4().hex}.wav"
    with open(tmp_file, "wb") as f:
        f.write(audio)
    segments, _ = await asyncio.to_thread(stt_model.transcribe, tmp_file, "es")
    texto = " ".join(seg.text for seg in segments)
    os.remove(tmp_file)
    return {"text": texto.strip()}

@app.get("/apps")
async def get_apps(): return APPS

@app.post("/apps")
async def save_apps(data: list = Body(...)):
    global APPS
    APPS = data
    try:
        with open("apps_config.json", "w", encoding="utf-8") as f:
            json.dump(APPS, f, ensure_ascii=False, indent=2)
        return {"ok": True}
    except Exception as e:
        return {"error": str(e)}

@app.post("/imagenes")
async def imagenes(data: dict = Body(...), cantidad: int = 18):
    tema = data.get("tema", "")
    if not tema:
        return {"error": "Debes indicar un tema"}
    offset = random.randint(0, 50)
    url = f"https://www.bing.com/images/search?q={tema}&first={offset}&FORM=HDRSC2"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.text, "html.parser")
    imgs = []
    for a in soup.find_all("a", {"class": "iusc"}):
        m_json = a.get("m")
        if m_json:
            try:
                m_data = json.loads(m_json)
                img_url = m_data.get("murl")
                if img_url:
                    imgs.append(img_url)
            except:
                continue
    random.shuffle(imgs)
    return {"imagenes": imgs[:cantidad]}

# ── HISTORY ──
@app.get("/history")
async def get_history():
    history = load_all_history()
    return [{"id": c["id"], "title": c["title"], "created": c["created"]} for c in reversed(history)]

@app.get("/history/{session_id}")
async def get_history_by_id(session_id: str):
    conv = get_conversation(session_id)
    if conv:
        return conv
    return {"error": "Conversación no encontrada"}

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    history = load_all_history()
    history = [c for c in history if c["id"] != session_id]
    save_all_history(history)
    if session_id in HISTORIA:
        del HISTORIA[session_id]
    return {"ok": True}

@app.patch("/history/{session_id}")
async def rename_history(session_id: str, request: Request):
    data = await request.json()
    new_title = data.get("title", "").strip()
    if not new_title:
        return {"error": "Título vacío"}
    history = load_all_history()
    for conv in history:
        if conv["id"] == session_id:
            conv["title"] = new_title
            save_all_history(history)
            return {"ok": True}
    return {"error": "No encontrada"}

@app.post("/history/new")
async def new_session():
    sid = str(uuid.uuid4())
    return {"session_id": sid}

@app.post("/buscar")
async def buscar_endpoint(data: dict = Body(...)):
    query = data.get("query", "")
    if not query:
        return {"error": "Debes indicar una consulta"}
    results = await asyncio.to_thread(buscar_web, query)
    return {"results": results}

# ── PROMPT EDITOR ──
@app.get("/prompt")
async def get_prompt():
    global PROMPT
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            PROMPT = f.read()
    except:
        pass
    return {"prompt": PROMPT}

@app.post("/prompt")
async def save_prompt(request: Request):
    global PROMPT
    data = await request.json()
    PROMPT = data.get("prompt", "")
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(PROMPT)
    return {"ok": True}

# ── FILE UPLOAD ──
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 20_000_000:
        return {"error": "Archivo demasiado grande (max 20MB)"}
    filepath = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{file.filename}")
    with open(filepath, "wb") as f:
        f.write(content)
    text = extract_text_from_file(filepath, file.filename)
    return {"filename": file.filename, "text": text[:8000], "path": filepath}

# ── RAG ──
@app.post("/rag/index")
async def rag_index(data: dict = Body(...)):
    folder = data.get("folder", "")
    if not folder or not os.path.isdir(folder):
        return {"error": "Carpeta no encontrada"}

    all_chunks = []
    file_count = 0
    for root, dirs, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            text = extract_text_from_file(fpath, fname)
            if text and not text.startswith("Tipo de archivo") and not text.startswith("Error"):
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append({"text": chunk, "source": fpath})
                file_count += 1

    if not all_chunks:
        return {"error": "No se encontraron archivos indexables"}

    # Generate embeddings
    logging.info(f"RAG: Indexando {len(all_chunks)} chunks de {file_count} archivos...")
    embeddings = []
    for i in range(0, len(all_chunks), 10):
        batch = [c["text"] for c in all_chunks[i:i+10]]
        try:
            resp = await asyncio.to_thread(ollama.embed, model="nomic-embed-text", input=batch)
            embeddings.extend(resp["embeddings"])
        except Exception as e:
            logging.error(f"RAG embed error: {e}")
            return {"error": f"Error generando embeddings: {e}"}

    index = {
        "chunks": [c["text"] for c in all_chunks],
        "sources": [c["source"] for c in all_chunks],
        "embeddings": embeddings
    }
    save_rag_index(index)
    logging.info(f"RAG: Indexación completa. {len(all_chunks)} chunks.")
    return {"ok": True, "files": file_count, "chunks": len(all_chunks)}

@app.get("/rag/status")
async def rag_status():
    index = load_rag_index()
    return {"chunks": len(index.get("chunks", [])), "has_index": len(index.get("chunks", [])) > 0}

# ── LOGS WEBSOCKET ──
@app.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket):
    await websocket.accept()
    log_buffer.clients.append(websocket)
    try:
        # Send existing logs
        for entry in log_buffer.logs[-50:]:
            await websocket.send_json(entry)
        # Keep alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in log_buffer.clients:
            log_buffer.clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")