"""
AmbedkarGPT-Intern-Task
-----------------------

Simple command-line Q&A system using:

- LangChain (for RAG orchestration)
- Chroma (local vector store)
- HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
- Ollama LLM with mistral 7B

The system:
1. Loads 'speech.txt'
2. Splits it into chunks
3. Creates embeddings and stores them in a local Chroma vector store
4. Retrieves relevant chunks based on user questions
5. Uses an LLM (Ollama + Mistral) to answer questions from that context

Run:
    python main.py
"""

import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


SPEECH_FILE = "speech.txt"


def load_documents(file_path: str = SPEECH_FILE):
    """
    Load the speech from a plain text file as LangChain Documents.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Make sure it exists in the project root."
        )

    # TextLoader loads the text file and wraps it into a list of Document objects
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def split_documents(documents):
    """
    Split the document into manageable chunks.
    We use CharacterTextSplitter which splits by characters with some overlap.
    """
    # You can tune chunk_size and chunk_overlap as needed
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    """
    Create a Chroma vector store using HuggingFace embeddings.
    This is an in-memory store for this simple prototype.
    """
    # Use sentence-transformers/all-MiniLM-L6-v2 as required
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create the Chroma vector store from documents
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def create_qa_chain(vector_store):
    """
    Create a RetrievalQA chain that:
    - Uses the Chroma vector store as retriever
    - Uses Ollama (Mistral) as the LLM to generate answers
    """
    # This assumes you have Ollama running locally and have pulled the 'mistral' model:
    #   ollama pull mistral
    llm = Ollama(model="mistral")

    # RetrievalQA chain:
    # - chain_type="stuff" simply stuffs retrieved docs into the prompt.
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True,  # helpful for debugging / transparency
        chain_type="stuff",
    )
    return qa


def run_cli(qa_chain):
    """
    Simple command-line loop that:
    - Takes user questions
    - Runs them through the RetrievalQA chain
    - Prints the answers

    Type 'exit', 'quit', or 'q' to terminate.
    """
    print("=" * 70)
    print("AmbedkarGPT - Q&A on 'Annihilation of Caste' Excerpt")
    print("Ask a question about the speech. Type 'exit' to quit.")
    print("=" * 70)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Assistant: Goodbye! 👋")
            break

        if not query:
            print("Assistant: Please type a question or 'exit' to quit.")
            continue

        try:
            # The RetrievalQA chain expects a dict with the key "query"
            result = qa_chain({"query": query})

            answer = result.get("result", "")
            sources = result.get("source_documents", [])

            print("\nAssistant:")
            print(answer)

            # (Optional) Show which chunks were used as context
            print("\n[Context Chunks Used:]")
            for i, doc in enumerate(sources, start=1):
                print(f"\n--- Chunk {i} ---")
                print(doc.page_content)

        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")
            print("Make sure Ollama is running and the 'mistral' model is available.")


def main():
    """
    Main entry point:
    1. Load documents from speech.txt
    2. Split them into chunks
    3. Build a Chroma vector store
    4. Create the RetrievalQA chain with Ollama (Mistral)
    5. Start the CLI loop
    """
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Number of chunks created: {len(chunks)}")

    print("Creating vector store (Chroma) with HuggingFace embeddings...")
    vector_store = create_vector_store(chunks)

    print("Creating RetrievalQA chain with Ollama (Mistral)...")
    qa_chain = create_qa_chain(vector_store)

    run_cli(qa_chain)


if __name__ == "__main__":
    main()
