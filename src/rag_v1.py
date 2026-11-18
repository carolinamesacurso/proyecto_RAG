#Hacer chunking recursivo de la documentacion. Debes elegir los parametros más adecuados con respecto al estilo de tus documentos.

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import logging
from pathlib import Path
from typing import List

import docx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

BASE_URL =os.getenv("GOOGLE_GEMINI_BASE_URL")
MODELO = "gemini-3.5-flash-lite"

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model=MODELO, api_key=API_KEY, base_url=BASE_URL, temperature=0)