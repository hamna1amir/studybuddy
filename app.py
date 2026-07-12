import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import platform
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path

load_dotenv()

st.set_page_config(page_title="Course Book Assistant", page_icon="📚")

# --- OCR tool paths (Windows local vs Linux/Streamlit Cloud) ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"D:\tesse\tesseract.exe"
    POPPLER_PATH = r"D:\Release-26.02.0-0\poppler-26.02.0\Library\bin"
else:
    POPPLER_PATH = None  # Linux server uses packages.txt-installed system tools

# --- Cached resources (load once) ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(name="course_books")

@st.cache_resource
def load_groq_client():
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

model = load_model()
collection = load_collection()
groq_client = load_groq_client()

# --- PDF processing functions (with OCR fallback) ---
def extract_pages_from_pdf(pdf_path, filename):
    reader = PdfReader(pdf_path)
    pages = []
    needs_ocr_pages = []

    # First pass: try normal text extraction
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"source": filename, "page_num": i + 1, "text": text})
        else:
            needs_ocr_pages.append(i + 1)

    # Second pass: OCR any pages that had no extractable text
    if needs_ocr_pages:
        st.sidebar.write(f"Running OCR on {len(needs_ocr_pages)} scanned page(s)...")
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        for page_num in needs_ocr_pages:
            img = images[page_num - 1]
            ocr_text = pytesseract.image_to_string(img)
            if ocr_text.strip():
                pages.append({"source": filename, "page_num": page_num, "text": ocr_text})

    pages.sort(key=lambda p: p["page_num"])
    return pages

def chunk_pages(pages, chunk_size=500, overlap=50):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page_num": page["page_num"]
            })
            start += chunk_size - overlap
    return chunks

def embed_and_add_to_db(chunks, filename):
    existing_count = collection.count()
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts).tolist()
        ids = [f"{filename}_{existing_count + i + j}" for j in range(len(batch))]
        metadatas = [{"source": c["source"], "page_num": c["page_num"]} for c in batch]
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

def is_already_embedded(filename):
    existing = collection.get(where={"source": filename}, limit=1)
    return len(existing["ids"]) > 0

# --- RAG functions ---
def search(question, n_results=5):
    query_embedding = model.encode([question]).tolist()
    return collection.query(query_embeddings=query_embedding, n_results=n_results)

def build_context(results):
    context_parts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        page = results["metadatas"][0][i]["page_num"]
        context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
    return "\n\n".join(context_parts)

def ask(question):
    results = search(question)
    context = build_context(results)
    prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in your course materials."
Always mention which source/page the answer came from.

Context:
{context}

Question: {question}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def get_chunks_by_page_range(source_file, start_page, end_page):
    results = collection.get(
        where={
            "$and": [
                {"source": source_file},
                {"page_num": {"$gte": start_page}},
                {"page_num": {"$lte": end_page}}
            ]
        }
    )
    combined = list(zip(results["documents"], results["metadatas"]))
    combined.sort(key=lambda x: x[1]["page_num"])
    return combined

def summarize_pages(source_file, start_page, end_page):
    chunks = get_chunks_by_page_range(source_file, start_page, end_page)
    if not chunks:
        return f"No content found for {source_file}, pages {start_page}-{end_page}."
    full_text = "\n\n".join([text for text, meta in chunks])
    max_chars = 15000
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
        note = "\n\n(Note: content truncated due to size.)"
    else:
        note = ""
    prompt = f"""Summarize the following course material clearly and concisely.
Organize the summary with the main topics covered.

Content from {source_file}, pages {start_page}-{end_page}:
{full_text}

Summary:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content + note

def get_available_sources():
    all_items = collection.get()
    sources = set(meta["source"] for meta in all_items["metadatas"])
    return sorted(sources)

# --- Sidebar: Upload ---
st.sidebar.header("📤 Upload a Book/Notebook")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    if is_already_embedded(uploaded_file.name):
        st.sidebar.info(f"'{uploaded_file.name}' is already in the library.")
    else:
        with st.sidebar.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            st.write("Extracting text...")
            pages = extract_pages_from_pdf(tmp_path, uploaded_file.name)

            st.write("Chunking...")
            chunks = chunk_pages(pages)

            st.write(f"Embedding {len(chunks)} chunks...")
            embed_and_add_to_db(chunks, uploaded_file.name)

            os.unlink(tmp_path)
            status.update(label=f"'{uploaded_file.name}' ready!", state="complete")
        st.sidebar.success(f"Added {len(chunks)} chunks from {uploaded_file.name}")

st.sidebar.divider()
st.sidebar.subheader("📖 Available Books")
sources = get_available_sources()
for s in sources:
    st.sidebar.text(f"• {s}")

# --- Main UI: Tabs for Chat and Summarize ---
st.title("📚 Course Book Assistant")

tab1, tab2 = st.tabs(["💬 Ask Questions", "📝 Summarize Pages"])

with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask a question about your course materials..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching course materials..."):
                answer = ask(question)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.subheader("Summarize a Page Range")
    sources = get_available_sources()
    if sources:
        selected_source = st.selectbox("Choose a book/file", sources)
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start page", min_value=1, value=1)
        with col2:
            end_page = st.number_input("End page", min_value=1, value=10)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarize_pages(selected_source, start_page, end_page)
            st.markdown(summary)
    else:
        st.info("Upload a book first to summarize it.")