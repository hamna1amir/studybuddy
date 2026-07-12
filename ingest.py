from pypdf import PdfReader
import os

def load_all_pdfs(data_folder="data"):
    """Reads every PDF in the data folder and returns text with source tracking."""
    all_pages = []
    pdf_files = [f for f in os.listdir(data_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"No PDF files found in '{data_folder}' folder!")
        return all_pages
    
    for filename in pdf_files:
        filepath = os.path.join(data_folder, filename)
        print(f"Reading: {filename}")
        reader = PdfReader(filepath)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                all_pages.append({
                    "source": filename,
                    "page_num": i + 1,
                    "text": text
                })
    return all_pages

def chunk_text(pages, chunk_size=500, overlap=50):
    """Splits page text into overlapping chunks, keeping track of source file."""
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

if __name__ == "__main__":
    pages = load_all_pdfs("data")
    print(f"\nLoaded {len(pages)} pages total")

    chunks = chunk_text(pages)
    print(f"Created {len(chunks)} chunks")
    print("\nFirst chunk example:")
    print(chunks[0])