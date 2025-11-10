# 🚀 LangChain Documentation Helper Bot

AI-powered assistant that answers technical questions directly from the official LangChain documentation using Retrieval-Augmented Generation (RAG).  
Built with **Python, Streamlit, OpenAI, and Pinecone/Chroma**.

---

## ✅ Overview

This project provides an interactive chatbot that lets users ask questions related to LangChain.  
Instead of hallucinating or guessing, the system retrieves real chunks of LangChain documentation,  
then generates accurate, source-grounded responses.

If a question is out-of-scope (ex: "How do I make pizza?")  
➡️ the bot responds with **"I don't know"** and shows **no sources**.

---

## ✅ ✨ Features

- ✅ Ask any LangChain-related question
- ✅ Answers grounded in real documentation (RAG)
- ✅ Shows sources **only when** relevant
- ✅ Rejects unrelated questions safely
- ✅ Chat history support (history-aware retrieval)
- ✅ Works with **Pinecone** or **Chroma** vector stores
- ✅ Easy to expand to other docsets

---

## ✅ 🧠 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| LLM | OpenAI Chat Models |
| Embeddings | `text-embedding-3-small` |
| Vector Store | ✅ Pinecone (preferred) or ✅ Chroma |
| Framework | LangChain |
| Docs Loader | `ReadTheDocsLoader` |
| Chunking | `RecursiveCharacterTextSplitter` |
| Environment | `.env` + `python-dotenv` |

---

## ✅ 🧩 System Architecture
User → Streamlit UI → run_llm() →
History-Aware Retriever → Vector Database (Pinecone/Chroma) →
StuffDocumentsChain → ChatOpenAI → Answer + Sources




Two main components:

- **ingestion.py** → Loads docs → splits → embeds → stores vectors  
- **core.py** → Builds RAG pipeline → retrieves → generates answer

---

## ✅ 🔧 Installation

```bash
git clone [(https://github.com/YanivBohbot/document_assitant_project/new/main)](https://github.com/YanivBohbot/document_assitant_project.git)
cd <project-folder>

pipenv install
pipenv shell

## Create .env

OPENAI_API_KEY=sk-xxxxxxx


## Run the App
streamlit run main.py
PINECONE_API_KEY=pcn-xxxxxxx
PINECONE_INDEX_NAME=langchain-docs
CHAT_MODEL=gpt-4o-mini
CHAT_TEMPERATURE=0

## Ingest the documentation (run once)
python ingestion.py
