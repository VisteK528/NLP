from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="qwen3-embedding")

vectorstore = Chroma(
    persist_directory="./vision_chroma_db",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 7,
        "fetch_k": 30,
    },
)

while True:
    query = input("\nEnter your query (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    print("Searching...")
    results = retriever.invoke(query)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i + 1} ---")
        print(f"Entity Name: {doc.metadata.get('entity_name', 'N/A')}")
        print(f"Signature: {doc.metadata.get('signature', 'N/A')}")
        print(f"Content snippet: {doc.page_content[:200]}...\n")
