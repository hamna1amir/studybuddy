import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="course_books")

groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def get_chunks_by_page_range(source_file, start_page, end_page):
    """Pulls ALL chunks from a specific file within a page range, in order."""
    results = collection.get(
        where={
            "$and": [
                {"source": source_file},
                {"page_num": {"$gte": start_page}},
                {"page_num": {"$lte": end_page}}
            ]
        }
    )
    # Pair up text with page numbers, then sort by page number
    combined = list(zip(results["documents"], results["metadatas"]))
    combined.sort(key=lambda x: x[1]["page_num"])
    return combined

def summarize_pages(source_file, start_page, end_page):
    chunks = get_chunks_by_page_range(source_file, start_page, end_page)
    
    if not chunks:
        return f"No content found for {source_file}, pages {start_page}-{end_page}. Check the filename and page numbers."
    
    full_text = "\n\n".join([text for text, meta in chunks])
    
    # Groq has a token limit per request - truncate if this range is huge
    max_chars = 15000  # roughly safe for a single request
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]
        truncated_note = "\n\n(Note: content was truncated because the page range was very large.)"
    else:
        truncated_note = ""

    prompt = f"""Summarize the following course material clearly and concisely. 
Organize the summary with the main topics covered.

Content from {source_file}, pages {start_page}-{end_page}:
{full_text}

Summary:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content + truncated_note

if __name__ == "__main__":
    print("Available files: book.pdf, digielec.pdf, Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf, ods-python.pdf, OOP.pdf")
    source = input("Which file? ")
    start = int(input("Start page: "))
    end = int(input("End page: "))
    
    print("\nSummarizing...")
    summary = summarize_pages(source, start, end)
    print(f"\n{summary}")