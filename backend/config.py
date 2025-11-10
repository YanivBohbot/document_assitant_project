
import os
from dotenv import load_dotenv

load_dotenv()

# Vector store (local fallback dir)
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL      = os.getenv("CHAT_MODEL", "gpt-4o-mini")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))

# External services
FIRECRAWL_KEY     = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY")

# Pinecone v3
PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "langchain-docs-idx")