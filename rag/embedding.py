import json
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

embeddings = OllamaEmbeddings(model="qwen3-embedding")

vision_libraries = ["opencv", "open3d", "pcl"]
documents = []

for lib in vision_libraries:
    file_path = f"data/parsed/{lib}_chunks.json"

    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping...")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        json_chunks = json.load(f)

    for chunk in tqdm(json_chunks, desc=f"Processing {lib} Documents"):
        page_content = chunk.get("entity_name", "") + " " + chunk.get("description", "")

        metadata = chunk.get("metadata", {}).copy()

        metadata["library"] = lib
        metadata["entity_name"] = chunk.get("entity_name", "")
        metadata["signature"] = chunk.get("signature", "")
        metadata["description"] = chunk.get("description", "")

        metadata["parameters"] = str(chunk.get("parameters", ""))
        metadata["returns"] = str(chunk.get("returns", ""))

        documents.append(Document(page_content=page_content, metadata=metadata))

if documents:
    print(f"Ingesting {len(documents)} total documents into Chroma...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./vision_chroma_db",
    )
    print("Data successfully ingested!")
else:
    print("No documents were found to ingest.")
