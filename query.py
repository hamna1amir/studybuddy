from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load your Groq API key from .env
load_dotenv()

# Set up the embedding model (same one used for storing)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="course_books")

# Set up Groq client (OpenAI-compatible)
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def search(question, n_results=5):
    """Search ChromaDB for the most relevant chunks to the question."""
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    return results

def build_context(results):
    """Turn search results into a readable context block with sources."""
    context_parts = []
    for i in range(len(results["documents"][0])):
        text = results["documents"][0][i]
        source = results["metadatas"][0][i]["source"]
        page = results["metadatas"][0][i]["page_num"]
        context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
    return "\n\n".join(context_parts)

def ask(question):
    """Full RAG pipeline: search, build context, ask Groq."""
    print("Searching course materials...")
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

if __name__ == "__main__":
    print("Course Book Q&A Assistant (type 'quit' to exit)\n")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "quit":
            break
        answer = ask(question)
        print(f"\nAnswer:\n{answer}\n")
        print("-" * 50)