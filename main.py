import os
# LangChain Community imports (Loaders, Stores, LLMs)
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# Modern LangChain imports (Chains & Prompts)
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


SPEECH_FILE = "speech.txt"


def load_documents(file_path: str = SPEECH_FILE):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Could not find '{file_path}'. Make sure it exists in the project root."
        )
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents


def split_documents(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks):
    # Suppress the deprecation warning in logs if you prefer, 
    # but this works fine for now.
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store


def create_qa_chain(vector_store):
    """
    Modern Chain Construction:
    1. Define Prompt
    2. Define Document Chain (LLM + Prompt)
    3. Define Retrieval Chain (Retriever + Document Chain)
    """
    llm = Ollama(model="mistral")

    # 1. Create a prompt template
    # The variable "{context}" is where the retrieved docs go.
    # The variable "{input}" is the user's question.
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the context below.
    If you don't know the answer, just say you don't know.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # 2. Create the document chain (stuffs docs into the context variable)
    document_chain = create_stuff_documents_chain(llm, prompt)

    # 3. Create the final retrieval chain
    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), document_chain)

    return retrieval_chain


def run_cli(qa_chain):
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
            continue

        try:
            # --- CRITICAL FIX HERE ---
            # Modern chains use .invoke()
            # The input key defined in our prompt is "input"
            result = qa_chain.invoke({"input": query})

            # The result dict contains:
            # "input": user query
            # "context": list of source documents
            # "answer": the LLM's response
            answer = result.get("answer", "")
            sources = result.get("context", [])

            print("\nAssistant:")
            print(answer)

            print("\n[Context Chunks Used:]")
            for i, doc in enumerate(sources, start=1):
                print(f"\n--- Chunk {i} ---")
                print(doc.page_content[:200] + "...") # Truncated for readability

        except Exception as e:
            print(f"\n[Error] Something went wrong: {e}")
            print("Make sure Ollama is running and the 'mistral' model is available.")


def main():
    print("Loading documents...")
    docs = load_documents()

    print("Splitting documents into chunks...")
    chunks = split_documents(docs)
    print(f"Number of chunks created: {len(chunks)}")

    print("Creating vector store (Chroma)...")
    vector_store = create_vector_store(chunks)

    print("Creating RetrievalQA chain with Ollama (Mistral)...")
    qa_chain = create_qa_chain(vector_store)

    run_cli(qa_chain)


if __name__ == "__main__":
    main()