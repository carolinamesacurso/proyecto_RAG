#Este archivo se encarga de preparar los datos para que el grafo pueda utilizarlos para responder preguntas

import os
import logging
from pathlib import Path
from typing import List
import docx
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
#Declaramos variables esenciales 

# Informas errores de lectura (archivos con formato incorrecto ...)
logger = logging.getLogger(__name__)

#Rutas de carpetas
BASE_DIR = Path(__file__).resolve().parent.parent
# __file__: identifica el archivo Python que se está ejecutando.
# .parent: sube de nivel en la estructura de carpetas (a src).
# Se utiliza para luego localizar docs y chroma.


DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_rag"

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx"}


#construye metadatos de cada documento
#Se utiliza para mostrar en el grafo difernte información al ususario (como de que documento viene)
def _meta(path: Path, **extra) -> dict:
    return {"source": str(path), "doc_key": path.name, **extra}

#Procesamos txt y md en una misma función
# intentamos leer el archivo en UTF-8
# Si produce error , probamos con cp1252 (documentos generados con windows)
# Devolvemos un objeto Document para que el langchain pueda procesarlo
def load_txt_md(path: Path) -> List[Document]:
    try:
        txt = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = path.read_text(encoding="cp1252", errors="replace")
    if not txt.strip():
        return []
    return [Document(page_content=txt, metadata=_meta(path))]

#Pocesamos pdf
#Comprobamos si hace falta una contraseña debido a que puede ser que esté cifrado , para ello primero probamso con una vacía
#Lo que permite trabajar con documento que no requieren contraseña de ususario
#Recorremos las páginas y creamso un objeto documents por cada una de ellas
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

#Procesamos docx
#Abrimos el documento y extraemos el texto
#Desarrollamos la posibilidad de que puedan contener tablas
#Unimos texto y tablas en una única variabkle para convertirlo en un document
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

#Relacionamos cada formato de documento con su función
LOADERS = {".txt": load_txt_md, ".md": load_txt_md, ".pdf": load_pdf, ".docx": load_docx}

#Obtiene la extensión del archivo y ejecuta la función correspondiente
#Si la extensión no es correcta , devuelve una lista vacía
def load_file(path: Path) -> List[Document]:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        return []
    try:
        return loader(path)
    except Exception as e:
        logger.warning("No se pudo leer %s: %s", path, e)
        return []

#Realizamos un chunking , el cual nos permite fragmental la información en trozos más pequeños
#Le pasamos nuestros documentos , el tamaño del chunk y el solapamiento
#El solapamiento permitirá que una idea que queda separada en dos fragmentos no pierda información compartida entre ellos
#chunk size nos indica el tamaño de los fragmentos
def chunkenizar(docs_dir: Path = DOCS_DIR, chunk_size: int = 300, chunk_overlap: int = 100) -> List[Document]:
    #El solapamiento tiene que ser mayor que el solapamiento , esto debido a que si no estaríamos generando fragmentos que tengan más información repetida que documentación original
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size")

    # Cargar todos los documentos
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
    #Configuración del splitter
    #RecursiveCharacterTextSplitter -> divide texto respetando separadores
    #Primero intenta dividirlo por ciertos separadores , si ve que son fragmentos grandes , utiliza unos más pequeños
    #add_start_index -> se añaden a los metadatos la posición de inicio de cada chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(raw_docs)
    #Asignamos un identificador único a cada chunk -> chunk_id
    #Chroma necesito un id por registro y no podemos tener duplicados
    for i, c in enumerate(chunks):
        pagina = f":p{c.metadata['page']}" if "page" in c.metadata else ""
        c.metadata["chunk_id"] = f"{Path(c.metadata['doc_key']).stem}{pagina}:c{i}"

    print(f"{len(files)} archivos -> {len(raw_docs)} documentos/páginas -> {len(chunks)} chunks")
    return chunks


chunks = chunkenizar(DOCS_DIR, chunk_size=60, chunk_overlap=20) #define cuántos caracteres o tokens se repiten entre el final de un fragmento (chunk) y el inicio del siguiente

#Preparamos chunks para poder transformarlos en vectores , unicamente trabajando con su contenido
textos = [c.page_content for c in chunks]

# Vemos los primeros chunks y sus metadatos
# for c in chunks30:
#     print(c.metadata["chunk_id"], "|", c.page_content[:80].replace("\n", " "))

# Cargamos modelo -> su función es convertir el texto en un valor numérico llamado embedding
# embedding = vector numérico que representa el contenido de un texto semánticamnete
modelo_emb = SentenceTransformer("BAAI/bge-m3")

# Con muchos chunks conviene ir por lotes y ver el progreso
#Generamos vectores
vectores = modelo_emb.encode(textos, normalize_embeddings=True, batch_size=16, show_progress_bar=True)
print("Forma de la matriz:", vectores.shape)   # (número de chunks, 1024)

#Guardamos colección fisicamente en chroma_rag y comprobamso si ya existe una colección llamada apuntes_rag 
#Si es el caso se elimina y creamos una nueva
# cliente = chromadb.PersistentClient(path="./chroma_rag")
cliente = chromadb.PersistentClient(path=str(CHROMA_DIR))

if "apuntes_rag" in [c.name for c in cliente.list_collections()]:
    cliente.delete_collection("apuntes_rag")

coleccion = cliente.create_collection(name="apuntes_rag", metadata={"hnsw:space": "cosine"})
#Preparamos metadatos 
#No incluimos chunk_id porque lo utilizamos como identificador del registro.
# Chroma solo admite metadatos str/int/float/bool: nos quedamos con los que nos interesan
def metadatos_chroma(c):
    m = {"source": c.metadata["source"], "doc_key": c.metadata["doc_key"],
         "start_index": c.metadata.get("start_index", 0)}
    if "page" in c.metadata:
        m["page"] = c.metadata["page"]
    return m

LOTE = 1000
#Guardamos los vectores en chroma con información adicional y metadatos
for i in range(0, len(chunks), LOTE):
    coleccion.add(
        ids=[c.metadata["chunk_id"] for c in chunks[i:i + LOTE]],
        embeddings=vectores[i:i + LOTE].tolist(),
        documents=textos[i:i + LOTE],
        metadatas=[metadatos_chroma(c) for c in chunks[i:i + LOTE]],
    )
print("Chunks guardados:", coleccion.count())
