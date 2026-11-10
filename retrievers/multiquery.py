from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import ChatOllama

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever()

llm = ChatOllama(
    model="llama3.2",
    temperature=0.7
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

query = "What is gradient descent?"

docs = multi_query_retriever.invoke(query)

print("\nRetrieved Documents:\n")

for doc in docs:
    print(doc.page_content)