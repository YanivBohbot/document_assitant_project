# main.py
from __future__ import annotations

from typing import Set

import streamlit as st
from backend.core import run_llm

st.set_page_config(page_title="LangChain – Documentation Helper", page_icon="📚", layout="wide")

st.header("LangChain – Documentation Helper Bot")

# Session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list[tuple(role, text)]

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = sorted(source_urls)
    lines = [f"{i+1}. {u}" for i, u in enumerate(sources_list)]
    return "sources:\n" + "\n".join(lines)

# Native chat input
user_prompt = st.chat_input("Ask something about the docs…")
if user_prompt:
    with st.spinner("Generating response…"):
        res = run_llm(query=user_prompt, chat_history=st.session_state["chat_history"])

        # Collect URLs from returned Documents
        srcs: Set[str] = set()
        for d in res.get("context", []):
            m = getattr(d, "metadata", {}) or {}
            u = m.get("source") or m.get("path") or m.get("url")
            if u:
                srcs.add(u)

        answer_block = f"{res['answer']}\n\n{create_sources_string(srcs)}"

        st.session_state["user_prompt_history"].append(user_prompt)
        st.session_state["chat_answers_history"].append(answer_block)
        st.session_state["chat_history"].append(("human", user_prompt))
        st.session_state["chat_history"].append(("ai", res["answer"]))

# Render history
for q, a in zip(st.session_state["user_prompt_history"], st.session_state["chat_answers_history"]):
    st.chat_message("user").write(q)
    st.chat_message("assistant").write(a)
