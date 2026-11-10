# 📚 PDF RAG Chatbot using Ollama, LangChain & ChromaDB

A local Retrieval-Augmented Generation (RAG) application built using:

- Ollama
- Llama 3.2
- Nomic Embeddings
- LangChain
- ChromaDB
- Streamlit

Users can upload PDF books and ask questions about the content. The application retrieves relevant information from the document and generates answers using a local LLM.

---

# 🚀 Features

✅ Upload any PDF

✅ Automatic text extraction

✅ Text chunking

✅ Vector embeddings

✅ ChromaDB vector storage

✅ Semantic search

✅ Question answering with Llama 3.2

✅ Fully local (No OpenAI API required)

✅ Streamlit UI

---

# 🛠 Tech Stack

Frontend:
- Streamlit

LLM:
- Ollama
- Llama 3.2

Embeddings:
- nomic-embed-text

Framework:
- LangChain

Vector Database:
- ChromaDB

Document Loader:
- PyPDFLoader

---

# 📂 Project Structure

RAG_Project
│
├── app.py
├── create_database.py
├── main.py
│
├── chroma_db/
│
├── document_loaders/
│   └── deep-learning.pdf
│
├── .venv/
│
└── README.md

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <repository-url>
cd RAG_Project
```

## Create Virtual Environment

```bash
uv venv
```

Activate Environment:

```bash
source .venv/Scripts/activate
```

---

# 📦 Install Dependencies

```bash
uv pip install streamlit langchain langchain-community langchain-text-splitters langchain-ollama chromadb pypdf
```

---

# 🤖 Install Ollama

Download from:

https://ollama.com/download

Verify installation:

```bash
ollama --version
```

---

# 📥 Download Required Models

## Llama 3.2

```bash
ollama pull llama3.2
```

## Embedding Model

```bash
ollama pull nomic-embed-text
```

Verify:

```bash
ollama list
```

Expected Output:

```text
llama3.2
nomic-embed-text
```

---

# 🏗 Workflow

```text
User Uploads PDF
        ↓
PyPDFLoader
        ↓
RecursiveCharacterTextSplitter
        ↓
Ollama Embeddings
        ↓
ChromaDB
        ↓
Retriever
        ↓
Llama 3.2
        ↓
Final Answer
```

---

# 📄 PDF Processing

```python
loader = PyPDFLoader(pdf_path)
docs = loader.load()
```

---

# ✂️ Text Chunking

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
```

---

# 🧠 Create Embeddings

```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

---

# 💾 Store Embeddings in ChromaDB

```python
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

---

# 🔍 Retrieval

```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k":4}
)
```

---

# 🤖 LLM Configuration

```python
llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)
```

---

# ▶️ Running Streamlit Application

```bash
streamlit run app.py
```

OR

```bash
uv run streamlit run app.py
```

---

# 🌐 Open Browser

Streamlit will generate:

```text
http://localhost:8501
```

Open the URL in your browser.

---

# 📖 How to Use

### Step 1

Upload a PDF

Example:

```text
deep-learning.pdf
```

### Step 2

Wait until:

```text
PDF processed successfully!
```

appears.

### Step 3

Ask Questions

Examples:

```text
What is Word2Vec?
```

```text
Explain CBOW.
```

```text
What is the Skip-Gram model?
```

```text
Summarize Chapter 8.
```

```text
Who introduced Word2Vec?
```

---

# 🐞 Common Errors and Fixes

## Ollama Connection Error

Error:

```text
connection refused
```

Fix:

```bash
ollama serve
```

---

## Model Not Found

Error:

```text
model not found
```

Fix:

```bash
ollama pull llama3.2
```

or

```bash
ollama pull nomic-embed-text
```

---

## PDF Not Loading

Error:

```text
FileNotFoundError
```

Verify PDF path:

```python
document_loaders/deep-learning.pdf
```

---

## OneDrive Installation Issues

Avoid creating projects inside:

```text
OneDrive/Desktop
```

Use:

```text
C:\Projects
```

instead.

---

# 📈 Future Improvements

- Chat History
- Multiple PDF Uploads
- Citation Sources
- Page Number References
- Conversational Memory
- Multiple Document Support
- Hybrid Search
- Re-ranking
- Dark / Light Themes
- Persistent Chat Sessions

---

# 🎯 Learning Outcomes

This project demonstrates:

- Document Loading
- Text Chunking
- Embeddings
- Vector Databases
- ChromaDB
- Semantic Search
- Retriever Design
- Prompt Engineering
- LangChain
- Ollama
- Streamlit
- Retrieval-Augmented Generation (RAG)

---

# 👨‍💻 Author

Nargis

Built while learning:

- LangChain
- Ollama
- ChromaDB
- Streamlit
- Retrieval-Augmented Generation (RAG)

---

# 🏆 Project Summary

This project allows users to upload PDF books and ask questions about the document contents. The system uses Ollama Embeddings to generate vector representations, stores them in ChromaDB, retrieves relevant chunks, and uses Llama 3.2 to answer questions from the uploaded book.

Fully Local ✅

No OpenAI API Required ✅

No Internet Required After Setup ✅

Powered by Ollama + LangChain + ChromaDB + Streamlit ✅