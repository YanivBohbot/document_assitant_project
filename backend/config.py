import os
from dotenv import load_dotenv
load_dotenv()
# ---- Config via env ----
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHAT_MODEL = os.getenv("OPENAI_API_KEY")
CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0"))
FIRECRAWL_KEY = os.getenv("FIRECRAWL_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")