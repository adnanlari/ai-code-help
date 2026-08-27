"""Streamlit placeholder - a thin HTTP client for the FastAPI backend.

Deliberately dumb: no business logic lives here. It only calls the backend over
HTTP and renders the JSON. That keeps the agent code fully decoupled from the
UI so this can be swapped for something else later without touching the backend.

Run:
    streamlit run frontend/app.py
(with the FastAPI server already up on :8000)
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Coding Buddy", page_icon="🧭")
st.title("AI Coding Buddy")
st.caption("Day 1 - indexing only. Chat + reasoning trace come on Day 3.")

# --- health indicator ---
try:
    h = requests.get(f"{BACKEND}/health", timeout=5).json()
    st.sidebar.success(f"backend: {h['status']} · db: {h['db']}")
except Exception as exc:  # noqa: BLE001
    st.sidebar.error(f"backend unreachable: {exc}")

# --- index a repo ---
st.subheader("Index a repository")
repo_url = st.text_input("GitHub URL", placeholder="https://github.com/pallets/click")
ref = st.text_input("ref (branch / tag / commit)", value="HEAD")

if st.button("Index", type="primary", disabled=not repo_url):
    with st.spinner("cloning, chunking, embedding..."):
        try:
            resp = requests.post(
                f"{BACKEND}/index",
                json={"repo_url": repo_url, "ref": ref or "HEAD"},
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"index failed: {exc}")
        else:
            if data["cached"]:
                st.info("Served from cache - no embedding calls made.")
            st.json(data)
            st.code(
                f'python -m scripts.query_vectors {data["repo_id"]} "how does X work?"',
                language="bash",
            )
