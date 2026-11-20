from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
from langgraph.graph import StateGraph, START, END

BASE_URL =os.getenv("GOOGLE_GEMINI_BASE_URL")
MODELO = "gemini-3.5-flash-lite"

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
# base_url=BASE_URL, -> al estar en clase
llm = ChatGoogleGenerativeAI(model=MODELO, api_key=API_KEY, temperature=0)
modelo_emb = SentenceTransformer("BAAI/bge-m3") 
BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_rag"

cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))

coleccion = cliente.get_collection(name="apuntes_rag")

class Estado(TypedDict):
    pregunta: str
    documentos: list[dict]   # los chunks que encuentra la búsqueda
    respuesta: str           # lo que contesta el LLM

K = 3   # cuántos chunks le pasamos al LLM

def recuperar(estado: Estado) -> dict:
    vector_pregunta = modelo_emb.encode([estado["pregunta"]], normalize_embeddings=True).tolist()
    res = coleccion.query(query_embeddings=vector_pregunta, n_results=K,
                          include=["documents", "metadatas", "distances"])

    encontrados = []
    for doc_id, texto, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        fuente = meta["doc_key"] + (f", pág. {meta['page']}" if "page" in meta else "")
        encontrados.append({"id": doc_id, "texto": texto, "fuente": fuente, "similitud": round(1 - dist, 2)})

    return {"documentos": encontrados}

INSTRUCCIONES = """Eres un asistente que responde preguntas usando ÚNICAMENTE los documentos proporcionados.
- Si la respuesta no está en los documentos, di: "No tengo esa información en los documentos."
- Indica entre corchetes el id del documento que usas, tal como aparece, por ejemplo [apuntes:p3:c12].
- Responde en el mismo idioma en que está escrita la pregunta, de forma breve."""

def generar(estado: Estado) -> dict:
    contexto = "\n\n".join(f"[{d['id']}] (fuente: {d['fuente']})\n{d['texto']}" for d in estado["documentos"])

    mensajes = [
        ("system", INSTRUCCIONES),
        ("human", f"DOCUMENTOS:\n{contexto}\n\nPREGUNTA: {estado['pregunta']}"),
    ]
    respuesta = llm.invoke(mensajes)
    return {"respuesta": respuesta.content}


grafo = StateGraph(Estado)

grafo.add_node("recuperar", recuperar)
grafo.add_node("generar", generar)

grafo.add_edge(START, "recuperar")
grafo.add_edge("recuperar", "generar")
grafo.add_edge("generar", END)

rag = grafo.compile()

def preguntar(pregunta):
    estado_final = rag.invoke({"pregunta": pregunta})

    print(f" {pregunta}\n")
    print("Chunks recuperados:")
    for d in estado_final["documentos"]:
        print(f"   {d['similitud']:.2f} | [{d['id']}] ({d['fuente']}) {d['texto'][:100].replace(chr(10), ' ')}")
    print(f"\n{estado_final['respuesta']}")
    print("-" * 80)



#chat de inputs con while true
if __name__ == "__main__":

    while True:
        pregunta = input("Pregunta>>> ")

        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("Chat terminado")
            break

        preguntar(pregunta)