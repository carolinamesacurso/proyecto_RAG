from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("No se ha encontrado GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

print("Enviando pregunta a Gemini...")

respuesta = llm.invoke("Di solamente: Hola, funciona")

print("Respuesta:", respuesta.content)