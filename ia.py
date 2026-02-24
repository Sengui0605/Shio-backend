# ia.py
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import random, json, requests
from bs4 import BeautifulSoup
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# -----------------------------
# Inicializar Firebase
# -----------------------------
cred = credentials.Certificate("ruta/a/tu/clave-firebase.json")  # Tu JSON de Firebase
firebase_admin.initialize_app(cred)
db = firestore.client()

# -----------------------------
# Configuración FastAPI
# -----------------------------
app = FastAPI(title="Shio AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia a tu dominio para seguridad
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Helpers
# -----------------------------
def guardar_historia(user_id, role, content):
    doc_ref = db.collection("usuarios").document(user_id)
    doc_ref.set({"user_id": user_id}, merge=True)
    doc_ref.collection("historia").add({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow()
    })

def obtener_historia(user_id, limit=50):
    doc_ref = db.collection("usuarios").document(user_id)
    historia_ref = doc_ref.collection("historia") \
                          .order_by("timestamp", direction=firestore.Query.DESCENDING) \
                          .limit(limit)
    return [{"role": doc.to_dict().get("role"), "content": doc.to_dict().get("content")}
            for doc in reversed(list(historia_ref.stream()))]

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/chat")
async def chat_post(data: dict = Body(...)):
    msg = data.get("msg", "").strip()
    user_id = data.get("user_id", "__anon__")
    
    if not msg:
        return {"text": "No enviaste ningún mensaje"}

    guardar_historia(user_id, "user", msg)

    # Respuesta simple de IA
    respuesta = f"Shio: recibí tu mensaje -> {msg}"

    guardar_historia(user_id, "assistant", respuesta)

    return {"text": respuesta}

@app.get("/historia")
async def historia(user_id: str):
    return {"historia": obtener_historia(user_id)}