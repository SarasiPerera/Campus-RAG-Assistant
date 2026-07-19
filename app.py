"""
University Regulations RAG Assistant — Streamlit app.

Same RAG pipeline as the notebook (chunk -> embed -> ChromaDB -> retrieve ->
Groq generation), wrapped in a shareable chat interface.

Run locally:   streamlit run app.py
Deploy free:   https://streamlit.io/cloud  (see README.md)
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions
import streamlit as st
from groq import Groq

MODEL = "llama-3.3-70b-versatile"
KB_PATH = "knowledge_base/student_handbook.txt"

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the \
University of Sri Jayewardenepura Student Handbook, using only the provided context \
passages. Answer clearly and concisely. If the answer is not contained in the context, \
say so explicitly rather than guessing or using outside knowledge. Cite the handbook \
section name when relevant."""


# ---------------------------------------------------------------------------
# RAG pipeline (identical logic to the notebook)
# ---------------------------------------------------------------------------

def load_and_chunk(path, chunk_size=800, overlap=100):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    sections = re.split(r"={10,}\n(.+?)\n={10,}", text)
    chunks = []
    current_header = "Introduction"
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if part.isupper() and len(part) < 100:
            current_header = part
            continue
        if len(part) <= chunk_size:
            chunks.append({"text": part, "section": current_header})
        else:
            start = 0
            while start < len(part):
                end = start + chunk_size
                chunks.append({"text": part[start:end], "section": current_header})
                start = end - overlap
    return chunks


@st.cache_resource(show_spinner="Indexing the student handbook...")
def build_index():
    chunks = load_and_chunk(KB_PATH)
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection("student_handbook")
    except Exception:
        pass
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = chroma_client.create_collection(
        name="student_handbook", embedding_function=embedding_fn
    )
    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        metadatas=[{"section": c["section"]} for c in chunks],
    )
    return collection


def retrieve(collection, query, k=4):
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "section": meta["section"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def ask(client, collection, question, k=4):
    retrieved = retrieve(collection, question, k=k)
    context = "\n\n---\n\n".join(
        f"[Section: {r['section']}]\n{r['text']}" for r in retrieved
    )
    user_message = f"Context passages from the Student Handbook:\n\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content, retrieved


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Campus RAG Assistant", page_icon="🎓", layout="centered")

st.title("🎓 Campus RAG Assistant")
st.caption(
    "Ask questions about the University of Sri Jayewardenepura Student Handbook "
    "(22nd edition, 2018). Answers are generated only from retrieved passages "
    "of the actual handbook — a Retrieval-Augmented Generation (RAG) demo."
)

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input(
        "Groq API key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Free at console.groq.com/keys — no card required.",
    )
    st.divider()
    st.markdown(
        "**How this works**\n\n"
        "1. The handbook is split into chunks\n"
        "2. Your question is matched against the most relevant chunks (ChromaDB)\n"
        "3. Those chunks + your question are sent to an LLM (Groq / Llama 3.3), "
        "instructed to answer only from the provided text\n\n"
        "[Project README](README.md) \u00b7 "
        "[Source PDF](https://www.sjp.ac.lk/pdf_uploads/Student_Hand_Book_Engilsh.pdf)"
    )
    st.divider()
    st.markdown(
        "**Known limitation:** the disciplinary punishment table (Schedule II) "
        "didn't extract cleanly from the source PDF and is excluded — "
        "questions about specific punishment tariffs will correctly come back "
        "as *not found* rather than a guess."
    )

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

collection = build_index()
client = Groq(api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources used"):
                for r in msg["sources"]:
                    st.markdown(f"**{r['section']}** (distance: {r['distance']:.3f})")
                    st.caption(r["text"][:300] + "...")

example_questions = [
    "What is the hostel curfew for female students?",
    "How does a student appeal a disciplinary decision?",
    "Who is eligible for a Mahapola scholarship?",
]
if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(len(example_questions))
    for col, q in zip(cols, example_questions):
        if col.button(q):
            st.session_state.pending_question = q

question = st.chat_input("Ask about the student handbook...")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the handbook..."):
            try:
                answer, sources = ask(client, collection, question)
            except Exception as e:
                answer = f"Something went wrong calling the API: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("Sources used"):
                for r in sources:
                    st.markdown(f"**{r['section']}** (distance: {r['distance']:.3f})")
                    st.caption(r["text"][:300] + "...")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
