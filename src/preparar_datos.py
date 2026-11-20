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
from sentence_transformers import SentenceTransformer
import chromadb

logger = logging.getLogger(__name__)
BASE_URL =os.getenv("GOOGLE_GEMINI_BASE_URL")
MODELO = "gemini-3.5-flash-lite"
#DOCS_DIR = Path.cwd() /docs

BASE_DIR = Path(__file__).resolve().parent.parent

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_rag"

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model=MODELO, api_key=API_KEY, base_url=BASE_URL, temperature=0)


def _meta(path: Path, **extra) -> dict:
    return {"source": str(path), "doc_key": path.name, **extra}


def load_txt_md(path: Path) -> List[Document]:
    try:
        txt = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = path.read_text(encoding="cp1252", errors="replace")
    if not txt.strip():
        return []
    return [Document(page_content=txt, metadata=_meta(path))]


def load_pdf(path: Path) -> List[Document]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")  # PDFs con contraseña vacía
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            docs.append(Document(page_content=txt, metadata=_meta(path, page=i + 1)))
    return docs


def load_docx(path: Path) -> List[Document]:
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for table in d.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    txt = "\n".join(parts).strip()
    if not txt:
        return []
    return [Document(page_content=txt, metadata=_meta(path))]


LOADERS = {".txt": load_txt_md, ".md": load_txt_md, ".pdf": load_pdf, ".docx": load_docx}


def load_file(path: Path) -> List[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    try:
        return loader(path)
    except Exception as e:
        logger.warning("No se pudo leer %s: %s", path, e)
        return []


def chunkenizar(docs_dir: Path = DOCS_DIR, chunk_size: int = 300, chunk_overlap: int = 100) -> List[Document]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    docs_d = Path(docs_dir).resolve()
    files = sorted(
        p for p in docs_d.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_EXTS
        and not p.name.startswith(("~$", "."))   # ignora temporales de Word y ocultos
    )
    if not files:
        logger.warning("No hay documentos en %s (pdf/docx/txt/md).", docs_d)
        return []

    raw_docs: List[Document] = []
    for p in files:
        raw_docs.extend(load_file(p))
    if not raw_docs:
        logger.warning("No se pudo extraer texto útil de los documentos.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)

    for i, c in enumerate(chunks):
        pagina = f":p{c.metadata['page']}" if "page" in c.metadata else ""
        c.metadata["chunk_id"] = f"{Path(c.metadata['doc_key']).stem}{pagina}:c{i}"

    print(f"{len(files)} archivos -> {len(raw_docs)} documentos/páginas -> {len(chunks)} chunks")
    return chunks


chunks30 = chunkenizar(DOCS_DIR, chunk_size=30, chunk_overlap=10) #define cuántos caracteres o tokens se repiten entre el final de un fragmento (chunk) y el inicio del siguiente

textos = [c.page_content for c in chunks30]

# Vemos los primeros chunks y sus metadatos
# for c in chunks30:
#     print(c.metadata["chunk_id"], "|", c.page_content[:80].replace("\n", " "))


modelo_emb = SentenceTransformer("BAAI/bge-m3")   # la primera vez tarda: descarga ~2,3 GB

# Con muchos chunks conviene ir por lotes y ver el progreso
vectores = modelo_emb.encode(textos, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
print("Forma de la matriz:", vectores.shape)   # (número de chunks, 1024)


cliente = chromadb.PersistentClient(path="./chroma_rag")

if "apuntes_rag" in [c.name for c in cliente.list_collections()]:
    cliente.delete_collection("apuntes_rag")

coleccion = cliente.create_collection(name="apuntes_rag", metadata={"hnsw:space": "cosine"})

# Chroma solo admite metadatos str/int/float/bool: nos quedamos con los que nos interesan
def metadatos_chroma(c):
    m = {"source": c.metadata["source"], "doc_key": c.metadata["doc_key"],
         "start_index": c.metadata.get("start_index", 0)}
    if "page" in c.metadata:
        m["page"] = c.metadata["page"]
    return m

LOTE = 1000
for i in range(0, len(chunks30), LOTE):
    coleccion.add(
        ids=[c.metadata["chunk_id"] for c in chunks30[i:i + LOTE]],
        embeddings=vectores[i:i + LOTE].tolist(),
        documents=textos[i:i + LOTE],
        metadatas=[metadatos_chroma(c) for c in chunks30[i:i + LOTE]],
    )
print("Chunks guardados:", coleccion.count())
