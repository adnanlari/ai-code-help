"""Streamlit UI - a thin HTTP client for the FastAPI backend.

No business logic here: it indexes repos and asks questions over HTTP and renders
the JSON. Display formatting lives in frontend/render.py (pure, tested). The app
can be swapped for anything else without touching backend/agent code.

Run (backend must already be up on :8000):
    streamlit run frontend/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `frontend.render` resolves
from frontend.render import (  # noqa: E402
    citation_line,
    grounding_summary,
    meta_line,
    retrieved_line,
    trace_line,
)

BACKEND = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Coding Buddy", page_icon="🧭", layout="wide")


# --- HTTP helpers ---------------------------------------------------------


def _err_detail(r: requests.Response) -> str:
    ctype = r.headers.get("content-type", "")
    if ctype.startswith("application/json"):
        try:
            return str(r.json().get("detail", r.text))
        except ValueError:
            pass
    return r.text[:300]


def api_get(path: str, timeout: float = 10):
    try:
        r = requests.get(f"{BACKEND}{path}", timeout=timeout)
    except requests.RequestException as exc:
        return False, str(exc)
    return (True, r.json()) if r.ok else (False, f"{r.status_code}: {_err_detail(r)}")


def api_post(path: str, payload: dict, timeout: float = 600):
    try:
        r = requests.post(f"{BACKEND}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return False, str(exc)
    return (True, r.json()) if r.ok else (False, f"{r.status_code}: {_err_detail(r)}")


# --- rendering ----------------------------------------------------------


def render_answer(resp: dict) -> None:
    st.markdown(resp.get("answer") or "_(empty answer)_")
    st.markdown("**Grounding:** " + grounding_summary(resp))

    cites = resp.get("citations") or []
    if cites:
        with st.expander(f"Citations ({len(cites)})", expanded=not resp.get("grounded")):
            for c in cites:
                st.markdown(citation_line(c))

    trace = resp.get("trace") or []
    if trace:
        with st.expander(f"Reasoning trace — {len(trace)} tool call(s)"):
            for step in trace:
                st.markdown(trace_line(step))
                if step.get("result_preview"):
                    st.code(step["result_preview"])

    retrieved = resp.get("retrieved") or []
    if retrieved:
        with st.expander(f"Retrieved chunks — the RAG seed ({len(retrieved)})"):
            for ch in retrieved:
                st.markdown(retrieved_line(ch))

    st.caption(meta_line(resp))


# --- sidebar: health + indexing + knobs -------------------------------

st.session_state.setdefault("history", [])
st.session_state.setdefault("repo", None)

with st.sidebar:
    st.subheader("Backend")
    ok, data = api_get("/health")
    if ok:
        st.success(f"{data['status']} · db {data['db']}")
    else:
        st.error(f"unreachable — {data}")

    st.subheader("Repository")
    url = st.text_input("GitHub URL", placeholder="https://github.com/pallets/click")
    ref = st.text_input("ref (branch / tag / commit)", value="HEAD")
    if st.button("Index", type="primary", disabled=not url):
        with st.spinner("clone → filter → chunk → embed → store …"):
            ok, data = api_post("/index", {"repo_url": url, "ref": ref or "HEAD"})
        if ok:
            st.session_state.repo = data
            st.session_state.history = []
            st.success(("cache hit — " if data["cached"] else "") + f"{data['chunk_count']} chunks")
        else:
            st.error(data)

    repo = st.session_state.repo
    if repo:
        st.caption(f"active repo `{repo['repo_id']}`")
        st.caption(f"{repo['commit_sha'][:12]} · {repo['chunk_count']} chunks · {repo['status']}")

    st.subheader("Retrieval / loop")
    top_k = st.slider("top_k (chunks retrieved)", 1, 20, 5)
    max_iter = st.slider("max_iterations (tool-loop cap)", 1, 12, 6)


# --- main: chat -------------------------------------------------------

st.title("AI Coding Buddy 🧭")
st.caption(
    "Ask about the indexed repository. Each question is answered independently — "
    "there is no conversation memory yet."
)

repo = st.session_state.repo
if not repo:
    st.info("Index a repository in the sidebar to begin.")
    st.stop()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        render_answer(turn["resp"])

if question := st.chat_input("Ask about this codebase…"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        payload = {
            "repo_id": repo["repo_id"],
            "question": question,
            "top_k": top_k,
            "max_iterations": max_iter,
        }
        with st.spinner("retrieving + reasoning …"):
            ok, data = api_post("/ask", payload)
        if ok:
            render_answer(data)
            st.session_state.history.append({"q": question, "resp": data})
        else:
            st.error(data)
