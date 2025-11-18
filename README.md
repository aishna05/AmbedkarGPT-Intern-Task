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

## Setup & Installation

### 1. Clone the Repository
```bash
git clone (https://github.com/aishna05/AmbedkarGPT-Intern-Task.git)
cd AmbedkarGPT-Intern-Task
2. Set up Virtual Environment
Bash

# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt
4. Setup Ollama (LLM)
This project uses Ollama to run Mistral 7B locally.

Download Ollama from ollama.ai.

Pull the model:

ollama pull mistral
Ensure the Ollama app is running in the background.

How to Run
Simply execute the main script:

python main.py