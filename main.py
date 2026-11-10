from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
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

If the answer cannot be found in the context, respond exactly with:

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
    ]
)

print("RAG system created")
print("Press 0 to exit")

while True:

    query = input("You: ")

    if query == "0":
        break

    docs = retriever.invoke(query)

    if not docs:
        print("\nAI: No relevant documents found.")
        continue

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    print("\n========== RETRIEVED CONTEXT ==========\n")
    print(context)
    print("\n=======================================\n")

    final_prompt = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    response = llm.invoke(final_prompt)

    print(f"\nAI: {response.content}\n")