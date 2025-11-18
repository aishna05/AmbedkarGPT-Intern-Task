import os
import sys

# 1. Loaders and Splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# 2. Embeddings and Vector Store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 3. LLM and Chains
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration
SPEECH_FILE = "speech.txt"
MODEL_NAME = "mistral"  # The LLM model to use in Ollama

def main():
    print("--- Kalpit Pvt Ltd AI Intern Task: AmbedkarGPT ---")
    
    # Step 1: Load the Document
    if not os.path.exists(SPEECH_FILE):
        print(f"Error: '{SPEECH_FILE}' not found. Please create it first.")
        sys.exit(1)
        
    print(f"1. Loading {SPEECH_FILE}...")
    loader = TextLoader(SPEECH_FILE, encoding="utf-8")
    docs = loader.load()

    # Step 2: Split Text
    print("2. Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)
    print(f"   > Created {len(chunks)} chunks.")

    # Step 3: Create Vector Store (Chroma)
    print("3. Creating Vector Store and Embeddings...")
    # Using all-MiniLM-L6-v2 as requested (runs locally, no API key)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create Chroma vector store in memory
    vector_store = Chroma.from_documents(chunks, embeddings)

    # Step 4: Setup LLM and RAG Chain
    print(f"4. Initializing Ollama with model '{MODEL_NAME}'...")
    llm = Ollama(model=MODEL_NAME)

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based ONLY on the provided context.
    
    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Create the chain: Document Chain (LLM+Prompt) -> Retrieval Chain (VectorStore+DocChain)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

    # Step 5: Interactive Loop
    print("\n✅ System Ready! Type 'exit' to quit.")
    
    while True:
        query = input("\nUser: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        if not query:
            continue

        print("Thinking...")
        try:
            # Invoke the chain
            response = retrieval_chain.invoke({"input": query})
            
            # Extract answer
            answer = response.get("answer", "No answer generated.")
            
            print(f"\nAssistant: {answer}")
            
            # Optional: Print source chunks for debugging/grading transparency
            # print("\n[Source Context used]")
            # for i, doc in enumerate(response.get("context", [])):
            #     print(f"Source {i+1}: {doc.page_content[:100]}...")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print(f"Tip: Ensure Ollama is running and you have pulled the model: 'ollama pull {MODEL_NAME}'")

if __name__ == "__main__":
    main()