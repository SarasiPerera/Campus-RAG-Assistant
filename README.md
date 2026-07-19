# Campus RAG Assistant

*A university's rulebook shouldn't require reading 60 pages to answer one question — this does it in one.*

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about a
university's official student handbook, grounded only in retrieved passages from the
real document — demoed here on the University of Sri Jayewardenepura's Student
Handbook, using semantic search combined with an LLM for grounded answer generation.

## Why this project

Built to demonstrate the core skills most 2026 "AI Engineer" internship postings screen
for specifically: RAG pipelines, vector databases, and LLM API integration — distinct
from classical ML (model training/evaluation), which is covered in other projects in this
portfolio.

## How it works

```
User question
     |
     v
Embed question (local ONNX MiniLM model, via ChromaDB)
     |
     v
Semantic search over indexed handbook chunks (ChromaDB vector store)
     |
     v
Top-k relevant passages retrieved
     |
     v
Passages + question sent to an LLM (Groq API, Llama 3.3 70B) with instructions to
answer ONLY from the provided context
     |
     v
Grounded answer returned
```

## Data source

`knowledge_base/student_handbook.txt` — text extracted from the official University of
Sri Jayewardenepura Student Handbook (22nd amended edition, September 2018), publicly
hosted at https://www.sjp.ac.lk/pdf_uploads/Student_Hand_Book_Engilsh.pdf

## Tech stack

- **ChromaDB** — vector database, with its built-in local embedding model (no cost, runs
  on-device, no data leaves your machine for the retrieval step)
- **Groq API (Llama 3.3 70B)** — answer generation, grounded in retrieved context; free,
  no card required, and very fast inference
- **Python / Jupyter** — everything runs interactively, cell by cell

The generation step is provider-agnostic by design — swapping in the Anthropic API,
OpenAI, or another provider only touches the `ask()` function, not the retrieval pipeline.

## How to run

### Option A — Streamlit app (recommended, shareable)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens a chat interface in your browser at `localhost:8501`. Paste your Groq API key
into the sidebar to start. Includes example question buttons and a "Sources used"
panel under each answer showing exactly which handbook passages it was grounded in.

**To get a live, shareable link** (worth doing for your CV/LinkedIn):
1. Push this project to a public GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub
3. Click "New app", select this repo and `app.py` as the entry point
4. Deploy — free, takes about a minute, gives you a public URL like
   `your-app-name.streamlit.app`
5. **Do not put your API key in the repo.** In Streamlit Cloud's app settings, add it
   under "Secrets" instead, or just have visitors paste their own key in the sidebar
   (the app supports both)

### Option B — Jupyter notebook (step-by-step / exploratory)

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `rag_university_handbook.ipynb` and run cells top to bottom — useful for
inspecting each stage of the pipeline (chunking, retrieval, generation) individually.

Both options share the exact same RAG logic — the notebook is better for understanding
and demonstrating each step; the app is better for a live demo or sending someone a link.

## Known limitations

- **Schedule II (the detailed disciplinary punishment table) is excluded from the
  knowledge base.** It's rendered as a formatted table in the source PDF that didn't
  extract cleanly to text. A production version would need OCR or manual transcription
  to cover it — documented here rather than silently dropped or included as garbled text.
- **The handbook is the 2018 edition** — the most recent version publicly hosted by the
  university at the time this project was built. Some details may have since changed.
- **Section-based chunking**, not semantic chunking — works well for this
  clearly-structured document but would need a different strategy for less structured
  sources.

## Possible extensions

- Add OCR to recover Schedule II
- Add conversation memory so follow-up questions resolve against prior turns
- Swap the local embedding model for a hosted one (e.g. Voyage AI) for better retrieval
  quality on more ambiguous queries
- Generalize beyond USJP: swap in any institution's handbook to reuse the same pipeline

## Related projects

- [E-Commerce SQL Analytics](../sql_analytics_project) — SQL-focused companion project
  (RFM segmentation, cohort retention, window functions) for Data Analyst/Data Science
  applications
