#load
#split into chunks
#create the embediings
#store into chroma

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

data = PyPDFLoader("document_loaders/deep-learning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

chunks = splitter.split_documents(docs)

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)