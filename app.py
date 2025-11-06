# """
# Personalized RAG — Streamlit app
# - Upload PDF / DOCX / TXT (<= 20 MB each)
# - Chunk -> embed (sentence-transformers) -> store in ChromaDB (local persist)
# - Retrieve top-k chunks -> generate answer with HF seq2seq model (flan-t5 by default)
# """

# import os
# import hashlib
# from pathlib import Path
# from typing import List, Dict

# import streamlit as st
# import chromadb
# from chromadb.config import Settings
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# # LangChain text splitter for robust chunking
# # from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Document loaders (langchain-community)
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# # -----------------------
# # Configuration
# # -----------------------
# st.set_page_config(page_title="Personalized RAG (Chroma)", layout="wide")
# st.title("📚 Personalized RAG — Document Chat (Chroma)")
# st.markdown("Upload docs (PDF / DOCX / TXT ≤ 20 MB each), index them, and ask questions backed by your documents.")

# UPLOAD_DIR = "uploaded_docs"
# PERSIST_DIR = "chroma_db"
# COLLECTION_NAME = "rag_collection"
# MAX_FILE_MB = 20

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(PERSIST_DIR, exist_ok=True)

# # -----------------------
# # Utilities
# # -----------------------
# def sha1_file(path: str) -> str:
#     h = hashlib.sha1()
#     with open(path, "rb") as f:
#         while True:
#             chunk = f.read(8192)
#             if not chunk:
#                 break
#             h.update(chunk)
#     return h.hexdigest()

# def load_file_to_docs(path: str) -> List[Dict]:
#     """
#     Load file and return list of dicts: {"text": ..., "meta": {...}}
#     """
#     ext = Path(path).suffix.lower()
#     docs = []
#     try:
#         if ext == ".pdf":
#             loader = PyPDFLoader(path)
#             pages = loader.load()
#             for i, p in enumerate(pages):
#                 text = p.page_content or ""
#                 if text.strip():
#                     docs.append({"text": text, "meta": {"source": Path(path).name, "page": i}})
#         elif ext == ".docx":
#             loader = Docx2txtLoader(path)
#             pages = loader.load()
#             for i, p in enumerate(pages):
#                 text = p.page_content or ""
#                 if text.strip():
#                     docs.append({"text": text, "meta": {"source": Path(path).name}})
#         elif ext == ".txt":
#             loader = TextLoader(path, encoding="utf-8")
#             pages = loader.load()
#             for i, p in enumerate(pages):
#                 text = p.page_content or ""
#                 if text.strip():
#                     docs.append({"text": text, "meta": {"source": Path(path).name}})
#         else:
#             st.warning(f"Unsupported file type: {ext}")
#     except Exception as e:
#         st.error(f"Failed to load {path}: {e}")
#     return docs

# def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
#     """
#     Character-based chunking with overlap.
#     """
#     text = text.replace("\r\n", "\n")
#     if len(text) <= chunk_size:
#         return [text]
#     chunks = []
#     start = 0
#     L = len(text)
#     while start < L:
#         end = min(start + chunk_size, L)
#         chunk = text[start:end].strip()
#         if chunk:
#             chunks.append(chunk)
#         if end == L:
#             break
#         start = end - overlap
#     return chunks

# # -----------------------
# # Cached resources
# # -----------------------
# @st.cache_resource(show_spinner=False)
# def get_chroma_client_and_collection(persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
#     client = chromadb.Client(Settings(persist_directory=persist_dir))
#     try:
#         collection = client.get_collection(name=collection_name)
#     except Exception:
#         collection = client.create_collection(name=collection_name)
#     return client, collection

# @st.cache_resource(show_spinner=False)      
# def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#     return SentenceTransformer(model_name)

# @st.cache_resource(show_spinner=False)
# def get_generator_pipeline(model_name: str = "google/flan-t5-small"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
#     return gen

# # -----------------------
# # Ingestion function
# # -----------------------
# def ingest_files(file_paths: List[str], embedding_model, chunk_size: int = 800, chunk_overlap: int = 150) -> int:
#     """
#     Load files, chunk them, compute embeddings and upsert into Chroma collection.
#     Returns number of chunks ingested.
#     """
#     client, collection = get_chroma_client_and_collection()
#     ids, documents, metadatas, embeddings = [], [], [], []

#     for path in file_paths:
#         file_hash = sha1_file(path)
#         loaded_docs = load_file_to_docs(path)
#         for doc_idx, doc in enumerate(loaded_docs):
#             text = doc["text"]
#             meta_base = doc.get("meta", {})
#             chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
#             for c_idx, chunk in enumerate(chunks):
#                 doc_id = f"{file_hash}_{doc_idx}_{c_idx}"
#                 # skip if already exists
#                 try:
#                     existing = collection.get(ids=[doc_id])
#                     if existing and existing.get("ids"):
#                         continue
#                 except Exception:
#                     # get may throw if id not present, ignore
#                     pass
#                 ids.append(doc_id)
#                 documents.append(chunk)
#                 md = meta_base.copy()
#                 md.update({"chunk_index": c_idx})
#                 metadatas.append(md)

#     if not documents:
#         return 0

#     # embeddings in batches
#     batch_size = 256
#     for i in range(0, len(documents), batch_size):
#         batch = documents[i : i + batch_size]
#         embs = embedding_model.encode(batch, convert_to_numpy=True)
#         # convert to python lists
#         embeddings.extend(embs.tolist())

#     collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
#     return len(documents)

# # -----------------------
# # Retrieval function
# # -----------------------
# def retrieve(query: str, top_k: int, embedding_model) -> List[Dict]:
#     client, collection = get_chroma_client_and_collection()
#     q_emb = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
#     res = collection.query(query_embeddings=[q_emb], n_results=top_k, include=["documents", "metadatas", "distances"])
#     docs = res.get("documents", [[]])[0]
#     metas = res.get("metadatas", [[]])[0]
#     dists = res.get("distances", [[]])[0]
#     results = []
#     for d, m, dist in zip(docs, metas, dists):
#         results.append({"text": d, "meta": m, "distance": dist})
#     return results

# # -----------------------
# # Streamlit UI
# # -----------------------
# client, collection = get_chroma_client_and_collection()

# # Sidebar settings
# with st.sidebar:
#     st.header("Settings")
#     chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=800, step=100)
#     chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=150, step=50)
#     top_k = st.slider("Retriever top_k", min_value=1, max_value=10, value=3)
#     st.markdown("---")
#     st.write("Chroma DB dir:", os.path.abspath(PERSIST_DIR))
#     if st.button("Clear DB"):
#         try:
#             # Danger: removes persistent files
#             for root, dirs, files in os.walk(PERSIST_DIR):
#                 for f in files:
#                     os.remove(os.path.join(root, f))
#             st.success("Chroma DB files removed. Restart app to recreate.")
#         except Exception as e:
#             st.error(f"Failed to clear DB: {e}")

# col1, col2 = st.columns([2, 1])

# with col1:
#     st.subheader("Upload documents")
#     uploaded_files = st.file_uploader(
#         "Drag & drop or click to upload (pdf, docx, txt). Max 20 MB each.",
#         type=["pdf", "docx", "txt"],
#         accept_multiple_files=True
#     )

#     if uploaded_files:
#         saved_paths = []
#         too_large = []
#         for uf in uploaded_files:
#             size_mb = uf.size / (1024 * 1024)
#             if size_mb > MAX_FILE_MB:
#                 too_large.append((uf.name, size_mb))
#                 continue
            
#             # dest = os.path.join(UPLOAD_DIR, uf.name)
#             # with open(dest, "wb") as f:
#                 # f.write(uf.getbuffer())
#             # saved_paths.append(dest)
            
#             import tempfile

#             saved_paths = []
#             too_large = []
#             for uf in uploaded_files:
#                 size_mb = uf.size / (1024 * 1024)
#                 if size_mb > MAX_FILE_MB:
#                     too_large.append((uf.name, size_mb))
#                     continue

#                 # Save to a temporary file in memory (auto-deletes after session)
#                 suffix = Path(uf.name).suffix
#                 with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                     tmp.write(uf.getbuffer())
#                     tmp_path = tmp.name
#                     saved_paths.append(tmp_path)













#         if too_large:
#             for name, s in too_large:
#                 st.error(f"File '{name}' is too large: {s:.2f} MB (limit {MAX_FILE_MB} MB).")

#         if saved_paths:
#             st.info(f"Indexing {len(saved_paths)} file(s)...")
#             embedding_model = get_embedding_model()
#             ingested = ingest_files(saved_paths, embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#             st.success(f"Ingested {ingested} chunks into ChromaDB.")

#     st.markdown("---")
#     st.subheader("Ask a question")
#     question = st.text_input("Enter your question about uploaded documents:")
#     if st.button("Get Answer"):
#         if not question:
#             st.warning("Enter a question first.")
#         else:
#             embedding_model = get_embedding_model()
#             results = retrieve(question, top_k=top_k, embedding_model=embedding_model)
#             if not results:
#                 st.info("No relevant documents found in the vector store.")
#             else:
#                 st.markdown("### 🔎 Retrieved Chunks")
#                 for r in results:
#                     src = r["meta"].get("source", "unknown")
#                     chunk_idx = r["meta"].get("chunk_index", None)
#                     st.write(f"- Source: **{src}** | chunk: {chunk_idx} | distance: {r['distance']:.4f}")
#                     st.write(r["text"][:600] + "...")
#                 # build context
#                 context = "\n\n".join([r["text"] for r in results])
#                 prompt = f"Answer the question using ONLY the context below. If the answer is not present, say you don't have enough information.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

#                 gen = get_generator_pipeline()
#                 with st.spinner("Generating answer..."):
#                     out = gen(prompt, max_length=256, do_sample=False)
#                     # output handling differences across transformers versions
#                     answer = ""
#                     if isinstance(out, list) and len(out) > 0:
#                         answer = out[0].get("generated_text") or out[0].get("summary_text") or out[0].get("generated_text", "")
#                     elif isinstance(out, dict):
#                         answer = out.get("generated_text") or str(out)
#                     answer = answer.strip() if answer else "No answer generated."

#                 st.markdown("### ✅ Answer")
#                 st.info(answer)

# with col2:
#     st.markdown("## DB Status")
#     try:
#         cnt = collection.count()
#     except Exception:
#         cnt = 0
#     st.write("Stored chunks:", cnt)
#     st.write("Collection name:", COLLECTION_NAME)
#     st.write("Persist dir:", os.path.abspath(PERSIST_DIR))
#     st.markdown("---")
#     st.markdown("### Tips")
#     st.markdown("- Upload small clean files (≤20 MB).")
#     st.markdown("- Use PDF or DOCX for best formatting.")
#     st.markdown("- Increase chunk overlap to preserve context.")

"""
Optimized Personalized RAG — Streamlit app (improved)
Features & improvements:
- In-memory/temp upload (20 MB per file limit)
- Chunk -> embed (sentence-transformers) -> store in ChromaDB (persisted)
- Reuse cached resources: embedding model, generator pipeline, Chroma client
- Batched embedding with progress bar
- LangChain RecursiveCharacterTextSplitter for smarter chunks
- Streaming-style token-by-token UI update (simulated, word-by-word)
- Session-state for DB client & minimal writes
- Safe cleanup of temp files
"""

import os
import atexit
import tempfile
import shutil
import hashlib
from pathlib import Path
from typing import List, Dict

import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# ----------------------
# Config
# ----------------------
st.set_page_config(page_title="Personalized RAG (Optimized)", layout="wide")
st.title("📚 Personalized RAG — Document Chat (Optimized)")

# Directories & limits
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"
MAX_FILE_MB = 20

os.makedirs(PERSIST_DIR, exist_ok=True)

# ----------------------
# Utilities
# ----------------------
def sha1_file(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_file_to_docs(path: str) -> List[Dict]:
    """Return list of {'text':..., 'meta':{...}} for a file path."""
    ext = Path(path).suffix.lower()
    docs = []
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(path)
            pages = loader.load()
            for i, p in enumerate(pages):
                text = p.page_content or ""
                if text.strip():
                    docs.append({"text": text, "meta": {"source": Path(path).name, "page": i}})
        elif ext == ".docx":
            loader = Docx2txtLoader(path)
            pages = loader.load()
            for i, p in enumerate(pages):
                text = p.page_content or ""
                if text.strip():
                    docs.append({"text": text, "meta": {"source": Path(path).name}})
        elif ext == ".txt":
            loader = TextLoader(path, encoding="utf-8")
            pages = loader.load()
            for i, p in enumerate(pages):
                text = p.page_content or ""
                if text.strip():
                    docs.append({"text": text, "meta": {"source": Path(path).name}})
        else:
            st.warning(f"Unsupported file type: {ext}")
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
    return docs

# smarter chunker using LangChain helper (keeps sentence boundaries)
def chunk_documents_texts(texts: List[str], chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

# ----------------------
# Cached heavy resources (load once per session)
# ----------------------
@st.cache_resource(show_spinner=False)
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    # sentence-transformers model (fast, small)
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner=False)
def get_generator_pipeline(model_name: str = "google/flan-t5-small"):
    # small seq2seq model; change to your preferred model if you have GPU or API keys
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)

@st.cache_resource(show_spinner=False)
def get_chroma_client_and_collection(persist_dir: str = PERSIST_DIR, collection_name: str = COLLECTION_NAME):
    # initialize Chroma client and collection (persistent)
    client = chromadb.Client(Settings(persist_directory=persist_dir))
    try:
        coll = client.get_collection(name=collection_name)
    except Exception:
        coll = client.create_collection(name=collection_name)
    return client, coll

# Keep one Chroma client/collection in session_state for reuse
if "chroma" not in st.session_state:
    client, collection = get_chroma_client_and_collection()
    st.session_state["chroma"] = {"client": client, "collection": collection}
client = st.session_state["chroma"]["client"]
collection = st.session_state["chroma"]["collection"]

# ----------------------
# Ingestion logic (temp files, batching, progress)
# ----------------------
def ingest_paths(paths: List[str], embedding_model, chunk_size: int = 800, chunk_overlap: int = 150) -> int:
    """Ingest list of file paths into Chroma. Returns number of chunks added."""
    ids, documents, metadatas = [], [], []
    for path in paths:
        file_hash = sha1_file(path)
        loaded = load_file_to_docs(path)
        for doc_idx, item in enumerate(loaded):
            text = item["text"]
            meta = item.get("meta", {})
            chunks = chunk_documents_texts([text], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            for c_idx, chunk in enumerate(chunks):
                doc_id = f"{file_hash}_{doc_idx}_{c_idx}"
                # skip if exists
                try:
                    existing = collection.get(ids=[doc_id])
                    if existing and existing.get("ids"):
                        continue
                except Exception:
                    pass
                ids.append(doc_id)
                documents.append(chunk)
                md = meta.copy()
                md.update({"chunk_index": c_idx})
                metadatas.append(md)

    if not documents:
        return 0

    # batch embeddings with progress
    batch_size = 128
    all_embs = []
    progress = st.progress(0)
    total = len(documents)
    for i in range(0, total, batch_size):
        batch = documents[i : i + batch_size]
        embs = embedding_model.encode(batch, convert_to_numpy=True)
        all_embs.extend(embs.tolist())
        progress.progress(min(1.0, (i + batch_size) / total))
    progress.empty()

    # upsert
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=all_embs)
    return len(documents)

# ----------------------
# Retrieval & generator helpers
# ----------------------
def retrieve_top_k(query: str, k: int, embedding_model) -> List[Dict]:
    q_emb = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for d, m, dist in zip(docs, metas, dists):
        out.append({"text": d, "meta": m, "distance": dist})
    return out

def generate_answer_streaming(prompt: str, generator, chunk_delay: float = 0.02):
    """
    Simulate a streaming output by splitting generated result word-by-word
    and yielding incremental updates. This is safe for models that produce
    full output when called (like HF pipeline). For real streaming you'd
    use an API that supports streaming tokens.
    """
    out = generator(prompt, max_length=256, do_sample=False)
    text = ""
    if isinstance(out, list) and out:
        text = out[0].get("generated_text") or out[0].get("summary_text") or ""
    elif isinstance(out, dict):
        text = out.get("generated_text", "")
    else:
        text = str(out)
    # simple word-by-word streaming
    words = text.split()
    partial = ""
    for w in words:
        partial += w + " "
        yield partial
    # final
    yield partial.strip()

# ----------------------
# UI layout
# ----------------------
with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (chars)", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", min_value=0, max_value=500, value=150, step=50)
    top_k = st.slider("Retriever top_k", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.write("Chroma persist dir:", os.path.abspath(PERSIST_DIR))
    if st.button("Clear persistent DB"):
        # careful: remove entire folder to start fresh
        try:
            if os.path.exists(PERSIST_DIR):
                shutil.rmtree(PERSIST_DIR)
            # recreate empty client/collection
            client, collection = get_chroma_client_and_collection()
            st.session_state["chroma"] = {"client": client, "collection": collection}
            st.success("Cleared persistent DB. Reload the app.")
        except Exception as e:
            st.error(f"Failed to clear DB: {e}")

col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("Upload documents (drag & drop)")
    uploaded_files = st.file_uploader(
        "Drop PDF / DOCX / TXT files (max 20 MB each)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    temp_paths = []
    # Save temp files for processing (do not persist permanently)
    if uploaded_files:
        too_large = []
        for f in uploaded_files:
            size_mb = f.size / (1024 * 1024)
            if size_mb > MAX_FILE_MB:
                too_large.append((f.name, size_mb))
                continue
            suffix = Path(f.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(f.getbuffer())
            tmp.flush()
            tmp.close()
            temp_paths.append(tmp.name)
            # schedule cleanup at exit
            atexit.register(lambda p=tmp.name: os.remove(p) if os.path.exists(p) else None)

        if too_large:
            for name, s in too_large:
                st.error(f"File '{name}' too large: {s:.2f} MB (limit {MAX_FILE_MB} MB).")

        if temp_paths:
            st.info(f"Indexing {len(temp_paths)} file(s)... this may take a moment.")
            embedding_model = get_embedding_model()
            added = ingest_paths(temp_paths, embedding_model, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            st.success(f"Ingested {added} chunks into the vector store.")

    st.markdown("---")
    st.subheader("Ask a question")
    question = st.text_input("Enter your question about uploaded documents:")
    if st.button("Get Answer"):
        if not question:
            st.warning("Please enter a question first.")
        else:
            embedding_model = get_embedding_model()
            # retrieve top-k
            with st.spinner("Retrieving relevant chunks..."):
                results = retrieve_top_k(question, top_k, embedding_model)
            if not results:
                st.info("No relevant documents found.")
            else:
                st.markdown("#### Retrieved (top results):")
                for r in results:
                    src = r["meta"].get("source", "unknown")
                    idx = r["meta"].get("chunk_index", None)
                    st.write(f"- {src} (chunk {idx}) — distance {r['distance']:.4f}")
                    st.write(r["text"][:400] + "...")
                # build context & prompt
                context = "\n\n".join([r["text"] for r in results])
                prompt = f"Answer the question using ONLY the context below.Answer in appropriate format. If not present, say you don't have enough information.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                gen = get_generator_pipeline()
                st.markdown("### Answer (streaming)")
                placeholder = st.empty()
                # streaming simulation
                for partial in generate_answer_streaming(prompt, gen):
                    placeholder.info(partial)
                st.success("Done.")

with col2:
    st.markdown("## DB Status")
    try:
        cnt = collection.count()
    except Exception:
        cnt = 0
    st.write("Stored chunks:", cnt)
    st.write("Collection name:", COLLECTION_NAME)
    st.write("Persist dir:", os.path.abspath(PERSIST_DIR))
    st.markdown("---")
    st.markdown("### Tips")
    st.markdown("- Keep files small (<= 20 MB).")
    st.markdown("- Increase chunk overlap for better continuity.")
    st.markdown("- Use a GPU and larger models for higher quality generation.")
