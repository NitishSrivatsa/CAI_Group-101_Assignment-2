# RAG vs FT — Finance Q&A (Group 101: Multi‑Stage Retrieval + Supervised FT)

## Deploy to Streamlit Cloud
1. Create a GitHub repo and upload these files.
2. Put your cleaned 2023–2024 text files in `data/corpus/` (as `.txt`). Keep files small.
3. Add `data/qa_pairs.csv` (same ~50 Q/A for both systems) — template below.
4. On https://share.streamlit.io → New App → pick your repo → main file: `streamlit_app.py`.
5. In **App settings → Secrets**, optionally set:
   - `DATA_DIR = "data"`
   - `FT_ADAPTER_REPO = "<public-HF-repo-with-merged-weights>"`
   - `FT_ADAPTER_FILE = "<weights filename>"`
6. Deploy.

## Techniques implemented
- **RAG:** Multi‑Stage Retrieval (Stage 1: BM25 + Dense across small & large chunks; Stage 2: Cross‑Encoder re‑rank).
- **FT:** Supervised instruction path (FLAN‑T5). If merged fine‑tuned weights are provided on HF Hub, app will load them.

## Evaluation
Use the app to answer the 3 mandatory test questions + ≥10 extended questions. Copy the per‑row evaluation table from the UI or extend `evaluation.py` to export CSV.

## `data/qa_pairs.csv` template
```csv
question,answer
What was the company's revenue in 2023?,The company's revenue in 2023 was $4.13 billion.
What was the company's revenue in 2024?,The company's revenue in 2024 was $4.02 billion.
List key segments.,The key segments include Client, Data Center, and Network (example).
```
