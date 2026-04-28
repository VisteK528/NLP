import re

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


def get_library_filter(user_prompt: str) -> dict:
    match = re.search(r"\b(opencv|open[- ]?3d|pcl)\b", user_prompt, re.IGNORECASE)

    if match:
        raw_match = match.group(1).lower()
        normalized_lib = raw_match.replace(" ", "").replace("-", "")

        return {"library": normalized_lib}
    return {}


def get_library_docs_for_bm25_search(vectorstore: Chroma, batch_size: int):
    all_documents_raw = []
    all_metadatas_raw = []
    offset = 0

    while True:
        batch = vectorstore.get(
            include=["documents", "metadatas"], limit=batch_size, offset=offset
        )

        if not batch["documents"]:
            break

        all_documents_raw.extend(batch["documents"])
        all_metadatas_raw.extend(batch["metadatas"])
        offset += batch_size

    all_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(all_documents_raw, all_metadatas_raw)
    ]

    library_docs = {"opencv": [], "open3d": [], "pcl": []}
    return all_docs, library_docs

def retrieve_answer(question: str) -> str:
    dynamic_filter = get_library_filter(question)

    if dynamic_filter:
        print(f"Identified library filter: {dynamic_filter['library']}")
        dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": 7, "filter": dynamic_filter}
        )
        lib_key = dynamic_filter["library"]
        sparse_retriever = bm25_retrievers.get(lib_key) or bm25_retrievers["all"]
    else:
        print("No specific library identified.")
        dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 7})
        sparse_retriever = bm25_retrievers["all"]

    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever], weights=[0.5, 0.5]
    )

    retrieved_docs = ensemble_retriever.invoke(question)
    formatted_docs = []
    for i, doc in enumerate(retrieved_docs):
        entity = doc.metadata.get("entity_name", "Unknown Entity")
        signature = doc.metadata.get("signature", "N/A")
        parameters = doc.metadata.get("parameters", "None provided.")
        returns = doc.metadata.get("returns", "None provided.")

        description = doc.page_content

        doc_string = f"""--- Document {i + 1}: {entity} ---
        Signature:
        {signature}

        Description:
        {description}

        Parameters:
        {parameters}

        Returns:
        {returns}
        """
        formatted_docs.append(doc_string)

    context_string = "\n".join(formatted_docs)

    result = chain.invoke({"docs": context_string, "question": question})
    return result, retrieved_docs

embeddings = OllamaEmbeddings(model="qwen3-embedding")

vectorstore = Chroma(
    persist_directory="./vision_chroma_db",
    embedding_function=embeddings,
)

model = OllamaLLM(model="gemma2:9b", temperature=0.0)

template = """
You are an expert OpenCV, Open3D and PCL developer and technical documenter. Your job is to answer the user's question using ONLY the provided documentation context.

<rules>
1. STRICT GROUNDING: You must base your answer entirely on the provided `<context>`.
2. MISSING INFO: If the context does not contain the answer, reply exactly with: "I cannot find the answer to this in the retrieved documentation."
3. NO META-TALK: Never say "Based on the provided snippets" or "The context shows". Just state the facts directly.
4. CODE FORMATTING: Use markdown formatting for all C++ signatures, function names, and code blocks.
5. CODE EXAMPLES: If applicable, synthesize a short C++ or Python snippet showing how to call the function using the retrieved parameters.
</rules>

<context>
{docs}
</context>

Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

all_docs, library_docs = get_library_docs_for_bm25_search(vectorstore, batch_size=5000)

bm25_retrievers = {
    "all": BM25Retriever.from_documents(all_docs),
    "opencv": BM25Retriever.from_documents(library_docs["opencv"])
    if library_docs["opencv"]
    else None,
    "open3d": BM25Retriever.from_documents(library_docs["open3d"])
    if library_docs["open3d"]
    else None,
    "pcl": BM25Retriever.from_documents(library_docs["pcl"])
    if library_docs["pcl"]
    else None,
}

for retriever in bm25_retrievers.values():
    if retriever:
        retriever.k = 7



while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    result, retrieved_docs = retrieve_answer(question)
    print(result)
