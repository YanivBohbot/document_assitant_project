# backend/core.py
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from dotenv import load_dotenv

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Pinecone (optional at runtime)
try:
    from langchain_pinecone import PineconeVectorStore
    _HAS_PINECONE = True
except Exception:
    _HAS_PINECONE = False

from .config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    CHAT_MODEL,
    CHAT_TEMPERATURE,
    PINECONE_INDEX_NAME,
    PINECONE_API_KEY,
)

load_dotenv()

# Reuse a single embeddings instance
_EMB = OpenAIEmbeddings(model=EMBEDDING_MODEL)


def _to_sources(docs: Sequence[Document]) -> List[Dict[str, Any]]:
    """Compact, UI-friendly source list."""
    out: List[Dict[str, Any]] = []
    for d in docs:
        m = d.metadata or {}
        out.append(
            {
                "id": m.get("id") or m.get("source") or m.get("path"),
                "source": m.get("source") or m.get("path") or m.get("url"),
                "score": m.get("score"),
                "metadata": {k: v for k, v in m.items() if k != "score"},
                "preview": (d.page_content[:240] + "…")
                if d.page_content and len(d.page_content) > 260
                else d.page_content,
            }
        )
    return out


def _get_retriever(k: int = 4):
    """Prefer Pinecone if configured; else use local Chroma."""
    if _HAS_PINECONE and PINECONE_API_KEY and PINECONE_INDEX_NAME:
        vs = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=_EMB)
        return vs.as_retriever(search_kwargs={"k": k})
    vs = Chroma(persist_directory=CHROMA_DIR, embedding_function=_EMB)
    return vs.as_retriever(search_kwargs={"k": k})


def _build_history_aware_rag_chain(k: int = 4):
    """
    History-aware retriever + 'stuff' docs chain (Hub prompts).
    Input expects: {"input": str, "chat_history": list}
    """
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=CHAT_TEMPERATURE)
    retriever = _get_retriever(k=k)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    stuff_chain = create_stuff_documents_chain(llm, qa_prompt)
    hist_aware = create_history_aware_retriever(llm=llm, retriever=retriever, prompt=rephrase_prompt)
    return create_retrieval_chain(retriever=hist_aware, combine_docs_chain=stuff_chain)


def run_llm(query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Returns:
    {
      "answer": str,
      "context": List[Document],
      "sources": List[{ id, source, score, metadata, preview }],
      "input": str
    }
    """
    chat_history = chat_history or []
    chain = _build_history_aware_rag_chain(k=4)
    result = chain.invoke({"input": query, "chat_history": chat_history})
    docs: List[Document] = result.get("context", []) or []
    return {
        "answer": result.get("answer"),
        "context": docs,
        "sources": _to_sources(docs),
        "input": query,
    }
