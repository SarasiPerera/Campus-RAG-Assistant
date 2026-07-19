import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ------------------------------------------------------------------
md("""# CampusKnowledgeAssistant

A Retrieval-Augmented Generation (RAG) app that answers questions about the
**University of Sri Jayewardenepura Student Handbook** (22nd edition, 2018) using:

- **Chunking + local embeddings** to index the handbook
- **ChromaDB** as the vector store for semantic search
- **Claude (Anthropic API)** to generate grounded answers from retrieved passages

This demonstrates the core RAG pattern: retrieve relevant context first, then generate
an answer *constrained to that context* — rather than relying on the model's general
knowledge (which would not know this specific, current university's regulations).
""")

# ------------------------------------------------------------------
md("""## 1. Setup

You'll need a **Groq API key** — free, no card required. Get one at
[console.groq.com/keys](https://console.groq.com/keys) (sign in, then "Create API Key").
Groq hosts open-weight models (e.g. Llama, GPT-OSS) and serves them very fast, at no
cost for this kind of light usage.

**Never hardcode your API key in a notebook you'll push to GitHub.** This notebook reads
it from an environment variable or prompts securely with `getpass` if it isn't set.

*(Note: the RAG pipeline itself — chunking, embeddings, ChromaDB retrieval — is
generation-provider-agnostic. Swapping `MODEL`/the client below for the Anthropic API,
OpenAI, or any other provider only touches the `ask()` function in Section 5, nothing
upstream of it.)*""")

code("""import os
import re
import getpass

import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

client = Groq()
MODEL = "llama-3.3-70b-versatile"  # free on Groq, strong quality, very fast inference
print("Client ready.")
""")

# ------------------------------------------------------------------
md("""## 2. Load and Chunk the Document

The knowledge base is the university's official student handbook
(`knowledge_base/student_handbook.txt`, extracted from the
[official PDF](https://www.sjp.ac.lk/pdf_uploads/Student_Hand_Book_Engilsh.pdf)).

**Chunking strategy:** split on the document's own section headers (marked with `====`)
first, since those are natural semantic boundaries, then further split any long section
into ~800-character chunks with a small overlap so no chunk loses context at its edges.""")

code("""def load_and_chunk(path, chunk_size=800, overlap=100):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on the document's own section header markers first
    sections = re.split(r"={10,}\\n(.+?)\\n={10,}", text)
    # re.split with a capturing group interleaves: [preamble, header1, body1, header2, body2, ...]
    chunks = []
    current_header = "Introduction"
    for part in sections:
        part = part.strip()
        if not part:
            continue
        if part.isupper() and len(part) < 100:
            current_header = part
            continue
        # further split long sections into overlapping chunks
        if len(part) <= chunk_size:
            chunks.append({"text": part, "section": current_header})
        else:
            start = 0
            while start < len(part):
                end = start + chunk_size
                chunk_text = part[start:end]
                chunks.append({"text": chunk_text, "section": current_header})
                start = end - overlap
    return chunks


chunks = load_and_chunk("knowledge_base/student_handbook.txt")
print(f"Document split into {len(chunks)} chunks")
print(f"\\nExample chunk (section: {chunks[5]['section']}):\\n")
print(chunks[5]["text"][:400])
""")

# ------------------------------------------------------------------
md("""## 3. Embed and Index in ChromaDB

Uses ChromaDB's built-in local embedding model (a small ONNX-based sentence
transformer) — no API calls or cost for embedding, everything runs on-device.""")

code("""chroma_client = chromadb.Client()

# remove any existing collection from a previous run of this notebook
try:
    chroma_client.delete_collection("student_handbook")
except Exception:
    pass

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

collection = chroma_client.create_collection(
    name="student_handbook",
    embedding_function=embedding_fn,
)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    metadatas=[{"section": c["section"]} for c in chunks],
)

print(f"Indexed {collection.count()} chunks into ChromaDB")
""")

# ------------------------------------------------------------------
md("""## 4. Retrieval Function

Given a question, embed it and retrieve the top-k most semantically similar chunks.""")

code("""def retrieve(query, k=4):
    results = collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "section": meta["section"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


# quick sanity check
test_results = retrieve("What is the hostel curfew time?")
for r in test_results:
    print(f"[{r['section']}] (distance={r['distance']:.3f})")
    print(r["text"][:200].replace(chr(10), " "))
    print()
""")

# ------------------------------------------------------------------
md("""## 5. Generation — Answering with an LLM, Grounded in Retrieved Context

The retrieved chunks are inserted into the prompt, and the model is explicitly instructed
to answer **only** from the provided context, and to say so plainly if the answer isn't
in the retrieved passages — this is what keeps a RAG system honest instead of letting the
model fall back on ungrounded general knowledge.""")

code("""SYSTEM_PROMPT = '''You are a helpful assistant answering questions about the \
University of Sri Jayewardenepura Student Handbook, using only the provided context \
passages. Answer clearly and concisely. If the answer is not contained in the context, \
say so explicitly rather than guessing or using outside knowledge. Cite the handbook \
section name when relevant.'''


def ask(question, k=4, verbose=False):
    retrieved = retrieve(question, k=k)
    context = "\\n\\n---\\n\\n".join(
        f"[Section: {r['section']}]\\n{r['text']}" for r in retrieved
    )

    user_message = f'''Context passages from the Student Handbook:

{context}

Question: {question}'''

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    answer = response.choices[0].message.content

    if verbose:
        print("--- Retrieved sections ---")
        for r in retrieved:
            print(f"  - {r['section']} (distance={r['distance']:.3f})")
        print()

    return answer
""")

# ------------------------------------------------------------------
md("## 6. Demo Queries")

code("""print(ask("What is the curfew time for female students living in hostels?", verbose=True))""")

code("""print(ask("Who is eligible for a Mahapola scholarship and how many installments are paid per year?"))""")

code("""print(ask("If a student disagrees with a disciplinary decision, how do they appeal?"))""")

code("""# A question the handbook doesn't answer, to check the model says so rather than guessing
print(ask("What is the tuition fee for the MBBS programme?"))""")

# ------------------------------------------------------------------
md("""## 7. Interactive Q&A

Run this cell and type your own questions. Type `exit` to stop.""")

code("""while True:
    q = input("Ask about the student handbook (or 'exit'): ")
    if q.strip().lower() == "exit":
        break
    print()
    print(ask(q))
    print("\\n" + "-" * 60 + "\\n")
""")

# ------------------------------------------------------------------
md("""## Known Limitations

Documenting these honestly, in the same spirit as the query-optimization writeup in the
SQL/Postgres project:

- **Schedule II (the detailed punishment table for each disciplinary offence) is excluded
  from the knowledge base.** The source PDF renders this as a formatted table that did not
  extract cleanly to text — a real production system would need OCR or manual transcription
  to include it. Questions about specific punishment tariffs for specific offences will
  correctly come back as "not in the provided context" rather than a fabricated answer.
- **The handbook is the 2018 (22nd) edition** — the most recent version found publicly
  hosted by the university. Some administrative details (staff names, specific figures)
  may have changed since. This is disclosed here rather than presented as current.
- **Chunking is section-based, not semantic.** A more advanced version could use semantic
  chunking (splitting on topic shifts detected by the embedding model itself) rather than
  the document's own headers, which would help on documents with less clear structure.

## Possible Extensions

- Add OCR (e.g. via `pytesseract` on rendered PDF pages) to recover Schedule II
- Swap the local embedding model for a hosted one (e.g. Voyage AI, used natively by
  Anthropic) for better retrieval quality
- Add a simple Streamlit or FastAPI wrapper for a shareable chat interface instead of the
  in-notebook input loop
- Add conversation memory so follow-up questions ("what about male students?") resolve
  correctly against the prior turn
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("rag_university_handbook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
