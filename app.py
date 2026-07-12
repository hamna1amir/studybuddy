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
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

st.set_page_config(page_title="StudyBuddy", page_icon="🎓", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 900px;
    }
    .hero {
        background: linear-gradient(135deg, #FF6F61 0%, #B98CE8 100%);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 24px rgba(185, 140, 232, 0.25);
    }
    .hero h1 {
        color: white !important;
        font-size: 2.2rem;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        color: rgba(255,255,255,0.92);
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }
    section[data-testid="stSidebar"] {
        background-color: #F8F3FF;
    }
    section[data-testid="stSidebar"] .stButton button {
        border-radius: 10px;
    }
    .lib-chip {
        background: white;
        border: 1px solid #E8D9FF;
        border-radius: 10px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
        font-size: 0.9rem;
        color: #2D2A3E;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F3FF;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6F61 !important;
        color: white !important;
    }
    .stChatInput textarea {
        border-radius: 14px !important;
    }
    .stButton button {
        border-radius: 10px;
        font-weight: 600;
    }
    .summary-card {
        background: #FDFBFF;
        border: 1px solid #EEE3FF;
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 0.5rem;
    }
    [data-testid="stChatMessage"] {
        background: white;
        border: 1px solid #EEE3FF;
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    [data-testid="stChatMessage"] p:has(strong:first-child) {
        margin-top: 0.6rem;
        padding-top: 0.5rem;
        border-top: 1px dashed #E8D9FF;
        font-size: 0.82rem;
        color: #8B7A9E;
    }
    hr {
        margin: 0.6rem 0 !important;
        border-color: #EEE3FF !important;
    }
    .book-selector-label {
        font-size: 0.85rem;
        color: #8B7A9E;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- OCR tool paths ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"D:\tesse\tesseract.exe"
    POPPLER_PATH = r"D:\Release-26.02.0-0\poppler-26.02.0\Library\bin"
else:
    POPPLER_PATH = None

# --- Cached resources ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_or_create_collection(name="course_books")

@st.cache_resource
def load_groq_client():
    # Try Streamlit Cloud secrets first, fall back to local .env
    api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
    return OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

model = load_model()
collection = load_collection()
groq_client = load_groq_client()

# --- PDF processing ---
def extract_pages_from_pdf(pdf_path, filename):
    reader = PdfReader(pdf_path)
    pages = []
    needs_ocr_pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"source": filename, "page_num": i + 1, "text": text})
        else:
            needs_ocr_pages.append(i + 1)

    if needs_ocr_pages:
        st.sidebar.write(f"Running OCR on {len(needs_ocr_pages)} scanned page(s)...")
        images = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_PATH)

        def ocr_one_page(page_num):
            img = images[page_num - 1]
            text = pytesseract.image_to_string(img)
            return (page_num, text)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(ocr_one_page, needs_ocr_pages))

        for page_num, ocr_text in results:
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

# --- RAG functions (now with optional book filter) ---
def search(question, n_results=5, book_filter=None):
    query_embedding = model.encode([question]).tolist()
    if book_filter and book_filter != "All Books":
        return collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"source": book_filter}
        )
    return collection.query(query_embeddings=query_embedding, n_results=n_results)

def build_context(results):
    context_parts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        page = results["metadatas"][0][i]["page_num"]
        context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
    return "\n\n".join(context_parts)

def ask(question, book_filter=None):
    results = search(question, book_filter=book_filter)
    if not results["documents"][0]:
        return "I couldn't find that in the selected book. Try switching to 'All Books' or checking your spelling."
    context = build_context(results)
    prompt = f"""You are a helpful study assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find that in your course materials."

Format your response EXACTLY like this:
1. Write a clear, well-organized answer using the context (use bullet points if it helps clarity).
2. On a new line, add a separator: ---
3. On the next line, write: **Sources:** followed by a comma-separated list like [book.pdf, p.12], [otherbook.pdf, p.45]

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
st.sidebar.header("📤 Add a Book")
st.sidebar.caption("Drop in any PDF and StudyBuddy will learn it")
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
st.sidebar.subheader("📚 Your Library")
sources = get_available_sources()
if sources:
    for s in sources:
        st.sidebar.markdown(f'<div class="lib-chip">📖 {s}</div>', unsafe_allow_html=True)
else:
    st.sidebar.caption("No books yet — upload one above.")

# --- Main UI: Hero header ---
st.markdown("""
<div class="hero">
    <h1>🎓 StudyBuddy</h1>
    <p>Your friendly study companion — ask questions or get quick summaries from any book you upload</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["💬  Chat with StudyBuddy", "📝  Quick Summary"])

with tab1:
    sources = get_available_sources()
    book_options = ["All Books"] + sources

    st.markdown('<div class="book-selector-label">📖 Answer from</div>', unsafe_allow_html=True)
    selected_book = st.selectbox(
        "Answer from",
        book_options,
        label_visibility="collapsed",
        key="book_filter_select"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        avatar = "🧑" if msg["role"] == "user" else "🎓"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if question := st.chat_input("Ask StudyBuddy anything about your books..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🎓"):
            spinner_text = f"Searching {selected_book}..." if selected_book != "All Books" else "Searching your books..."
            with st.spinner(spinner_text):
                answer = ask(question, book_filter=selected_book)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

with tab2:
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.subheader("Get a Quick Summary")
    sources = get_available_sources()
    if sources:
        selected_source = st.selectbox("Choose a book/file", sources)
        col1, col2 = st.columns(2)
        with col1:
            start_page = st.number_input("Start page", min_value=1, value=1)
        with col2:
            end_page = st.number_input("End page", min_value=1, value=10)

        if st.button("✨ Summarize", use_container_width=True):
            with st.spinner("Summarizing..."):
                summary = summarize_pages(selected_source, start_page, end_page)
            st.markdown(summary)
    else:
        st.info("Upload a book first to summarize it.")
    st.markdown('</div>', unsafe_allow_html=True)