import streamlit as st
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")

st.title("📚 RAG PDF Chatbot")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp_file:

        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    with st.spinner("Processing PDF..."):

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300
        )

        chunks = splitter.split_documents(docs)

        # Embeddings
        embedding_model = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Vector DB
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )

        st.success("PDF processed successfully!")

        question = st.text_input(
            "Ask a question about the PDF"
        )

        if question:

            retrieved_docs = retriever.invoke(question)

            context = "\n\n".join(
                [doc.page_content for doc in retrieved_docs]
            )

            llm = ChatOllama(
                model="llama3.2",
                temperature=0.7
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are a helpful AI assistant.

Answer ONLY from the provided context.

If the answer cannot be found in the context,
respond:

I could not find the answer in the document.
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
                ]
            )

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            response = llm.invoke(final_prompt)

            st.subheader("Answer")
            st.write(response.content)

            with st.expander("Retrieved Context"):
                st.write(context)