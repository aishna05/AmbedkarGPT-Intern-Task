import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

SPEECH_FILE = "speech.txt"

def main():
    print("--- RAG RETRIEVAL TEST (No LLM Required) ---")
    
    # 1. Load Document
    if not os.path.exists(SPEECH_FILE):
        print(f"Error: {SPEECH_FILE} not found.")
        return
    
    print("1. Loading document...")
    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    docs = loader.load()
    
    # 2. Split Document
    print("2. Splitting text...")
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    print(f"   > Created {len(chunks)} chunks.")
    
    # 3. Create Vector Store
    print("3. Building Vector Store (using HuggingFace embeddings)...")
    # This downloads the embedding model (small, ~90MB) which usually works 
    # even when the main LLM servers are down.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embeddings)

    # 4. Search Loop
    print("\n✅ System Ready! Type a query to check if the code finds the right text.")
    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() in ["exit", "quit"]: break
            
        results = vector_store.similarity_search(query, k=2)
        
        print("\n--- Top Matching Text Found ---")
        for i, res in enumerate(results):
            print(f"[Chunk {i+1}]: \"{res.page_content[:200]}...\"")

if __name__ == "__main__":
    main()