# AmbedkarGPT-Intern-Task

Simple **command-line Q&A system** built as part of the **Kalpit Pvt Ltd – AI Intern Hiring: Assignment 1**.

The system performs **RAG (Retrieval-Augmented Generation)** over a short excerpt from Dr. B.R. Ambedkar’s *"Annihilation of Caste"*. It answers questions **only using the provided speech text**.

---

## Features / What This Prototype Does

1. **Loads** the provided `speech.txt` file.
2. **Splits** the text into manageable chunks using LangChain's `CharacterTextSplitter`.
3. **Creates embeddings** using **HuggingFaceEmbeddings** with  
   `sentence-transformers/all-MiniLM-L6-v2`.
4. **Stores embeddings** in a **local ChromaDB vector store** (in memory).
5. **Retrieves relevant chunks** for a user’s question.
6. **Generates answers** using **Ollama** with the **Mistral 7B** model.

All components are **local** and **free**:  
no API keys, no external cloud services.

---

## Tech Stack

- **Language**: Python 3.8+
- **Framework**: LangChain
- **Vector Store**: ChromaDB
- **Embeddings**: HuggingFace – `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: [Ollama](https://ollama.ai/) with **Mistral 7B** model
- **Interface**: Simple command-line (CLI)

---

## Project Structure

```text
AmbedkarGPT-Intern-Task/
│
├── main.py           # Main RAG pipeline + CLI
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── speech.txt        # Provided text (Ambedkar excerpt)
