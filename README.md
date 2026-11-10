# 📚 BookMind AI – PDF RAG Chatbot

A locally powered **Retrieval-Augmented Generation (RAG)** application that allows users to upload PDF books and interact with their content through natural language questions.

Built using **Python, LangChain, Ollama, ChromaDB, and Streamlit**, the application retrieves relevant passages from uploaded documents and uses a locally running LLM to generate context-aware answers.

Instead of relying on the model's general knowledge, BookMind AI grounds its responses in the retrieved document content.

---

## ✨ Features

- 📄 **PDF Upload:** Upload your own PDF books through the Streamlit interface.
- 📖 **Document Loading:** Extract text and metadata from PDF documents using PyPDFLoader.
- ✂️ **Text Chunking:** Split large documents into smaller, manageable chunks.
- 🧠 **Text Embeddings:** Generate vector representations using Nomic Embeddings.
- 💾 **Vector Storage:** Store document embeddings in ChromaDB.
- 🔍 **Semantic Retrieval:** Retrieve relevant document chunks using vector similarity and MMR.
- 🤖 **AI-Powered Question Answering:** Generate answers using Llama 3.2.
- 💬 **Interactive Chat Interface:** Ask questions and interact with your uploaded book.
- 🔒 **Local Processing:** Run the LLM and embedding models locally using Ollama.
- 🚫 **No OpenAI API Required:** No OpenAI API key or cloud-based LLM service is needed.

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend / UI | Streamlit |
| LLM | Llama 3.2 |
| Local Model Runtime | Ollama |
| Embedding Model | nomic-embed-text |
| LLM Framework | LangChain |
| Vector Database | ChromaDB |
| Document Loader | PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Retrieval Strategy | Maximum Marginal Relevance (MMR) |

---

## 📂 Project Structure

```text
RAG_Project/
│
├── main.py
├── create_database.py
├── chroma_db/
│
├── document_loaders/
│   └── deep-learning.pdf
│
├── .venv/
│
├── pyproject.toml
├── uv.lock
└── README.md
```

**File responsibilities:**

- `main.py` – Streamlit interface, PDF upload, document retrieval, and question answering.
- `create_database.py` – PDF loading, text splitting, embedding generation, and ChromaDB storage.
- `chroma_db/` – Persistent vector database containing processed document embeddings.
- `document_loaders/` – Example PDF documents for testing.

---

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG_Project
```

### 2. Create a Virtual Environment

Using UV:

```bash
uv venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows CMD:**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
uv add streamlit langchain-core langchain-community langchain-ollama langchain-text-splitters chromadb pypdf
```

### 4. Install Ollama

Download and install Ollama:

https://ollama.com/download

Verify the installation:

```bash
ollama --version
```

### 5. Download Required Models

**Llama 3.2 – Language Model**

```bash
ollama pull llama3.2
```

**Nomic Embed Text – Embedding Model**

```bash
ollama pull nomic-embed-text
```

Verify that both models are available:

```bash
ollama list
```

Expected models:

```text
llama3.2
nomic-embed-text
```

---

## 🏗️ RAG Architecture and Workflow

BookMind AI follows a two-stage RAG architecture:

1. Document Indexing
2. Retrieval and Answer Generation

### Stage 1: Document Indexing Pipeline

When a user uploads a PDF book, the application processes it and creates a searchable vector database.

```text
        User Uploads PDF
                |
                v
          PyPDFLoader
                |
                v
       Extract Document Text
                |
                v
  RecursiveCharacterTextSplitter
                |
                v
          Text Chunks
                |
                v
       Ollama Embeddings
       (nomic-embed-text)
                |
                v
       Generate Vectors
                |
                v
            ChromaDB
                |
                v
       Persistent Storage
```

**Step-by-step explanation:**

1. **Document Loading:** PyPDFLoader extracts text and metadata from the uploaded PDF.
2. **Text Splitting:** RecursiveCharacterTextSplitter divides the extracted text into smaller chunks.
3. **Embedding Generation:** Nomic Embeddings converts each chunk into a numerical vector representation.
4. **Vector Storage:** ChromaDB stores the generated embeddings along with their corresponding document content and metadata.

The resulting vector database is used for retrieving relevant information when the user asks a question.

### Stage 2: Retrieval and Generation Pipeline

When the user asks a question, the application searches the vector database for relevant document chunks.

```text
          User Question
                |
                v
       Query Embedding
                |
                v
       ChromaDB Retriever
                |
                v
        MMR Retrieval
                |
                v
    Retrieve Relevant Chunks
                |
                v
       Retrieved Context
                |
                v
       Prompt Construction
                |
                v
          Llama 3.2
                |
                v
       Context-Based Answer
                |
                v
        Streamlit Chat UI
```

**Step-by-step explanation:**

1. **User Query:** The user enters a question through the Streamlit chat interface.
2. **Query Embedding:** The question is converted into a vector using the same embedding model.
3. **Retrieval:** ChromaDB searches for relevant document chunks.
4. **MMR Search:** Maximum Marginal Relevance retrieves relevant and diverse chunks to reduce redundant context.
5. **Context Construction:** The retrieved text is combined with the user's question.
6. **Prompt Engineering:** A prompt instructs the LLM to answer only using the supplied context.
7. **Answer Generation:** Llama 3.2 generates a response using the retrieved information.
8. **Response Display:** The generated answer is displayed in the Streamlit interface.

---

## 🔧 Core Implementation

### 1. PDF Document Loading

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document_loaders/deep-learning.pdf")

docs = loader.load()
```

Loads the PDF and extracts its text into LangChain Document objects.

### 2. Text Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)
```

- `chunk_size=1000`: Maximum target size of each chunk.
- `chunk_overlap=200`: Preserves overlapping content between consecutive chunks.

Chunking helps the retriever find relevant sections instead of processing an entire book at once.

### 3. Embedding Generation

```python
from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)
```

Converts document text into vector representations that capture semantic meaning.

### 4. Vector Database Creation

```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)
```

Stores document chunks and their embeddings in ChromaDB for retrieval.

### 5. MMR-Based Retrieval

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)
```

**Retrieval parameters:**

- `k=4`: Returns four selected chunks.
- `fetch_k=10`: Initially considers ten candidate chunks.
- `lambda_mult=0.5`: Balances relevance and diversity.

MMR helps retrieve useful information while reducing repetitive context.

### 6. LLM Configuration

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0.2
)
```

Uses Llama 3.2 locally to generate answers based on retrieved document context.

### 7. Context-Based Prompting

The application uses a prompt that restricts the model to the retrieved document content.

```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer ONLY from the provided context.

        If the answer cannot be found in the context,
        respond exactly with:

        I could not find the answer in the document.

        Do not use your own knowledge.
        Do not guess.
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    )
])
```

This instruction helps reduce unsupported answers and keeps responses grounded in the retrieved context.

---

## ▶️ Running the Application

Make sure Ollama is installed and both required models have been downloaded.

Start the Streamlit application:

```bash
uv run streamlit run main.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

---

## 📖 How to Use

**Step 1 – Upload a PDF**

Open the application and upload a PDF book using the sidebar.

Example:

```text
deep-learning.pdf
```

**Step 2 – Build Knowledge Base**

Click the **Build Knowledge Base** button.

The application will:

- Load the PDF.
- Extract its text.
- Split the content into chunks.
- Generate embeddings.
- Store the vectors in ChromaDB.

**Step 3 – Start Chatting**

Once the knowledge base is ready, enter your question in the chat input.

Example questions:

```text
What is Word2Vec?
```

```text
Explain the CBOW architecture.
```

```text
What is the difference between CBOW and Skip-Gram?
```

```text
Explain the Skip-Gram model with an example.
```

```text
Summarize the main concepts of Chapter 8.
```

The application retrieves relevant passages and generates an answer using Llama 3.2.

---

## 🐞 Common Errors and Solutions

### 1. Ollama Connection Error

**Error:**

```text
Connection refused
```

**Solution:**

Make sure Ollama is running.

```bash
ollama serve
```

If Ollama is already running as a background service, you do not need to start another instance.

### 2. Model Not Found

**Error:**

```text
model not found
```

**Solution:**

Download the required model:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. PDF Loading Error

**Error:**

```text
FileNotFoundError
```

**Solution:**

Check the PDF path and ensure that the uploaded file is valid and accessible.

### 4. Embedding Model Error

**Solution:**

Verify that the embedding model is installed:

```bash
ollama list
```

If it is missing:

```bash
ollama pull nomic-embed-text
```

### 5. OneDrive Installation Issues

For fewer file synchronization and virtual environment issues, consider keeping your project outside OneDrive.

Example:

```text
C:\Projects\RAG_Project
```

---

## 🚀 Future Improvements

Potential enhancements planned for the project:

- [ ] Multiple PDF Upload Support
- [ ] Source Citations with Page Numbers
- [ ] Display Retrieved Document Chunks
- [ ] Conversational Memory
- [ ] Chat History Persistence
- [ ] Hybrid Search (Keyword + Semantic Search)
- [ ] Re-ranking Retrieved Documents
- [ ] Support for DOCX and EPUB Files
- [ ] OCR Support for Scanned PDFs
- [ ] Improved Document Management
- [ ] Dark and Light Themes

---

## 🎯 Learning Outcomes

Through this project, I explored and implemented important concepts in Generative AI and Retrieval-Augmented Generation:

- Retrieval-Augmented Generation (RAG)
- Document Loading and Processing
- Recursive Text Splitting
- Text Embeddings
- Vector Databases
- ChromaDB
- Semantic Search
- Maximum Marginal Relevance (MMR)
- Information Retrieval
- Prompt Engineering
- Local LLM Integration
- LangChain
- Ollama
- Streamlit Application Development

---

## 👩‍💻 Author

**Nargis**

Computer Science Engineering Student | Exploring Generative AI and AI Engineering

---

## 🏁 Project Summary

**BookMind AI** demonstrates how Retrieval-Augmented Generation can make PDF books interactive and searchable.

By combining LangChain, Ollama, Nomic Embeddings, and ChromaDB, the application creates a local document-question-answering system that retrieves relevant information from uploaded books and generates context-aware responses.

**Built with:** Python · LangChain · Ollama · ChromaDB · Streamlit

**Local LLM:** Llama 3.2

**Embedding Model:** nomic-embed-text

**No OpenAI API Required.**
