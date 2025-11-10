from __future__ import annotations
import os
from typing import List
from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec  # pinecone-client v3

from backend.config import (
    EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

load_dotenv()


def _ensure_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = {i["name"] for i in pc.list_indexes()}
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,    # embedding model’s dim (text-embedding-3-small = 1536)
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc


def ingest_docs() -> None:
    # 1) Load docs (ReadTheDocs local export or cloned dir)
    loader = ReadTheDocsLoader(path="https://docs.langchain.com/oss/python/langchain/agents")
    raw_documents: List[Document] = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    # 2) Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks")

    # 3) Fix sources to proper URLs
    for d in documents:
        old_path = d.metadata.get("source", "")
        # make an https:// URL out of the local path
        if old_path.startswith("langchain-docs"):
            d.metadata["source"] = old_path.replace("langchain-docs", "https:/")

    # 4) Embeddings and Pinecone index
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set")

    _ = _ensure_pinecone_index()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"Upserting {len(documents)} chunks into Pinecone index '{PINECONE_INDEX_NAME}'")
    PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print("✅ Added to Pinecone vector store")


if __name__ == "__main__":
    ingest_docs()
