import re

from langchain_chroma import Chroma
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


embeddings = OllamaEmbeddings(model="qwen3-embedding")

vectorstore = Chroma(
    persist_directory="./vision_chroma_db",
    embedding_function=embeddings,
)

model = OllamaLLM(model="gemma4:e4b", temperature=0.0)

template = """
You are an expert OpenCV, Open3D and PCL developer and technical documenter. Your job is to answer the user's question using ONLY the provided documentation context.

<rules>
1. STRICT GROUNDING: You must base your answer entirely on the provided `<context>`.
2. MISSING INFO: If the context does not contain the answer, reply exactly with: "I cannot find the answer to this in the retrieved documentation."
3. NO META-TALK: Never say "Based on the provided snippets" or "The context shows". Just state the facts directly.
4. CITATIONS REQUIRED: Every single factual claim or code snippet you provide MUST end with a citation to the specific document number it came from, formatted as [Doc X]. E.g., "The function requires a float [Doc 2]."
5. CODE FORMATTING: Use markdown formatting for all C++ signatures, function names, and code blocks.
6. CODE EXAMPLES: If applicable, synthesize a short C++ or Python snippet showing how to call the function using the retrieved parameters.
</rules>

<context>
{docs}
</context>

Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break

    dynamic_filter = get_library_filter(question)

    if dynamic_filter:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 7, "filter": dynamic_filter}
        )
    else:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    retrieved_docs = retriever.invoke(question)
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
    print(result)
