from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain import hub
from langchain_classic.chains.combine_documents import create_stuff_documents_chain # .chains.combine_documents 
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from const import INDEX_NAME
from .config import EMBEDDING_MODEL
load_dotenv()





def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    _embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docsearch = PineconeVectorStore(index_name=INDEX_NAME, embedding=_embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result