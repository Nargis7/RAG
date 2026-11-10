from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="Python is widely used in artificial intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in python.", metadata={"source": "Data_Science_book"}),
    Document(page_content="Natural language processing is a field of artificial intelligence.", metadata={"source": "NLP_book"})
]

embeddings_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)



result = vector_store.similarity_search("What is used for data analysis?", k=2)

for r in result:
    print(r.page_content)
    print(r.metadata)
    print()

retriver = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

docs = retriver.invoke("what is Natural Language Processing?")

for d in docs:
    print(d.page_content)
    # print(d.metadata)
    # print()