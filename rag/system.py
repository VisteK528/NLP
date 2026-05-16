import re

import ollama as _ollama
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM


_COMPARATIVE_RE = re.compile(
    r"\b(difference|differ|compare|comparison|vs\.?|versus|contrast|"
    r"when to use|which is better|pros and cons)\b",
    re.IGNORECASE,
)
_BETWEEN_RE = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:\?|$)", re.IGNORECASE
)

_SCORE_THRESHOLD = 0.5
_RERANKER_TOP_N = 3
_reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    top_n=_RERANKER_TOP_N,
)


def _bm25_text(meta: dict) -> str:
    parts = [
        meta.get("entity_name", ""),
        meta.get("signature", ""),
        meta.get("description", ""),
        meta.get("parameters", ""),
        meta.get("returns", ""),
    ]
    return " ".join(filter(None, parts))


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

    all_docs = []
    library_docs = {"opencv": [], "open3d": [], "pcl": []}

    for doc, meta in zip(all_documents_raw, all_metadatas_raw):
        bm25_doc = Document(page_content=_bm25_text(meta), metadata=meta)
        all_docs.append(bm25_doc)
        lib = meta.get("library", "")
        if lib in library_docs:
            library_docs[lib].append(bm25_doc)

    return all_docs, library_docs

def _retrieve_docs(query: str, lib_filter: dict, k: int) -> list:
    candidate_k = max(k * 3, 20)
    search_kwargs = {"k": candidate_k, "score_threshold": _SCORE_THRESHOLD}
    if lib_filter:
        search_kwargs["filter"] = lib_filter
    dense = vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs=search_kwargs
    )
    lib_key = lib_filter.get("library", "") if lib_filter else ""
    sparse = bm25_retrievers.get(lib_key) or bm25_retrievers["all"]
    sparse.k = candidate_k
    ensemble = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_reranker, base_retriever=ensemble
    )
    return compression_retriever.invoke(query)


def retrieve_answer(question: str) -> str:

    dynamic_filter = get_library_filter(question)

    if dynamic_filter:
        print(f"Identified library filter: {dynamic_filter['library']}")
    else:
        print("No specific library identified.")

    if _COMPARATIVE_RE.search(question):
        between = _BETWEEN_RE.search(question)
        if between:
            term_a = between.group(1).strip()
            term_b = between.group(2).strip()
            print(f"[comparative] split retrieval: '{term_a}' vs '{term_b}'")
            docs_a = _retrieve_docs(term_a, dynamic_filter, k=4)
            docs_b = _retrieve_docs(term_b, dynamic_filter, k=4)
            seen, retrieved_docs = set(), []
            for doc in docs_a + docs_b:
                key = doc.metadata.get("entity_name", doc.page_content[:60])
                if key not in seen:
                    seen.add(key)
                    retrieved_docs.append(doc)
        else:
            retrieved_docs = _retrieve_docs(question, dynamic_filter, k=7)
    else:
        retrieved_docs = _retrieve_docs(question, dynamic_filter, k=7)

    formatted_docs = []
    for i, doc in enumerate(retrieved_docs):
        entity = doc.metadata.get("entity_name", "Unknown Entity")
        signature = doc.metadata.get("signature", "N/A")
        parameters = doc.metadata.get("parameters", "None provided.")
        returns = doc.metadata.get("returns", "None provided.")

        description = doc.metadata.get("description") or doc.page_content

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

model = OllamaLLM(model="llama3.1:8b", temperature=0.0)

template = """
You are an expert OpenCV, Open3D and PCL developer and technical documenter. Your job is to answer the user's question using ONLY the provided documentation context.

<rules>
1. STRICT GROUNDING: Base your answer on the provided `<context>`. You may synthesize and compare information across multiple documents to answer questions about differences, similarities, or relationships — as long as every claim is supported by at least one retrieved document. Class names, typedef names, and parameter descriptions are valid evidence — use them to infer purpose and behavior when no explicit definition is given.
2. MISSING INFO: Only reply "I cannot find the answer to this in the retrieved documentation." when the context contains NO relevant information at all — not even indirect clues like type names or parameter semantics. Do NOT refuse when the context contains relevant pieces that together answer the question through synthesis or reasonable inference. Never withhold a context-supported answer because you suspect a different or "better" function exists outside the provided context — answer with what is present.
3. NO META-TALK: Never say "Based on the provided snippets" or "The context shows". Just state the facts directly.
4. CODE FORMATTING: Use markdown formatting for all C++ signatures, function names, and code blocks.
5. CODE EXAMPLES: Only include a code snippet if the context contains an explicit example or a complete enough signature to reconstruct the call exactly. Do NOT invent code from memory.
6. CONCISENESS: Be precise and direct. Do not restate the question. Lead with the answer 
   in a full sentence addressed to the user (e.g. "X is used for...", "X does..."). 
   Avoid filler phrases. Never copy raw documentation text verbatim as the answer.
7. NO OUTSIDE KNOWLEDGE: Do not add facts, behaviors, or examples that are not present in the provided context, even if you know them to be true.
8. CITATION: For every factual claim, mentally verify it appears in one of the documents above. If you cannot point to a specific document for a claim, omit the claim entirely.

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
