from sentence_transformers import SentenceTransformer
import chromadb
from ingest import load_all_pdfs, chunk_text

# Load the embedding model (downloads once, then cached)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load and chunk your PDFs (reusing the functions from ingest.py)
print("Loading and chunking PDFs...")
pages = load_all_pdfs("data")
chunks = chunk_text(pages)
print(f"Total chunks to embed: {len(chunks)}")

# Set up ChromaDB (this creates a local folder to store the database)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="course_books")

# Embed and store in batches (so it doesn't overwhelm memory)
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    texts = [c["text"] for c in batch]
    
    embeddings = model.encode(texts).tolist()
    
    ids = [f"chunk_{i+j}" for j in range(len(batch))]
    metadatas = [{"source": c["source"], "page_num": c["page_num"]} for c in batch]
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

print("\nDone! All chunks embedded and stored in ChromaDB.")
print(f"Total items in collection: {collection.count()}")