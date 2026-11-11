# IntelliHire — Explainable AI for Recruitment Decision Support

IntelliHire is a Streamlit recruiter dashboard backed by a Python AI pipeline. It parses job descriptions, PDF/DOCX/TXT resumes and candidate video introductions, ranks candidates with a transparent five-feature scoring equation, explains every score, and offers a grounded RAG assistant with controlled agentic workflows. Final hiring decisions always stay with the recruiter.

Implementation of the Major Project Phase-1 report *"IntelliHire: Explainable AI for Recruitment Decision Support"* (SMIT, Dept. of CSE, Group 30), using the exact technology stack specified there.

---

## Contents
1. [What maps to what](#1-what-maps-to-what)
2. [Architecture](#2-architecture)
3. [Quick start (local)](#3-quick-start-local)
4. [Using MySQL](#4-using-mysql)
5. [Configuring the LLM](#5-configuring-the-llm)
6. [Using the dashboard](#6-using-the-dashboard)
7. [How scoring works](#7-how-scoring-works)
8. [Security, privacy and ethics](#8-security-privacy-and-ethics)
9. [Testing](#9-testing)
10. [Deployment](#10-deployment)
11. [Project structure](#11-project-structure)
12. [Notes and limitations](#12-notes-and-limitations)

---

## 1. What maps to what

| Report item | Where it lives |
|---|---|
| **Stage 1** – multi-format ingestion + security validation | `intellihire/ingestion/validation.py` (extension allow-list, size limits, magic-byte checks, DOCX zip-bomb guard, filename sanitising, SHA-256 de-duplication, 100-file batch limit) |
| **Stage 2** – PyMuPDF / python-docx extraction, normalisation | `intellihire/ingestion/documents.py` (two-column reading order, hidden-text quarantine), `normalize.py` |
| Stage 2 – OpenCV + Whisper transcription | `intellihire/media/video.py`, `transcriber.py` (local Whisper or Whisper API) |
| Stage 2 – spaCy tokenisation / phrase matching | `intellihire/nlp/` – `spacy_pipeline.py`, `skills.py` (PhraseMatcher over a 140-skill alias taxonomy), `experience.py`, `jd_parser.py`, `resume_parser.py` |
| **Stage 3** – all-MiniLM-L6-v2 384-d vectors, cosine similarity | `intellihire/embeddings/encoder.py` (chunk + mean-pool so long resumes are not truncated) |
| **Stage 4** – S_SEMANTIC, S_EXPERIENCE, S_SKILL, S_VIDEO, S_COMM | `intellihire/scoring/features.py`, `media/communication.py` |
| Stage 4 – Scikit-learn & XGBoost classifiers, top-5 | `intellihire/scoring/ranker.py` (cross-validated LR / RF / XGBoost on recruiter decisions), top-5 cards on the Rankings page |
| **Stage 5** – MySQL + SQLAlchemy, FAISS, data isolation | `intellihire/db/models.py`, `db/session.py`, `rag/vector_store.py` (FAISS, optional ChromaDB) |
| Stage 5 – RAG explanation loop, isolated prompt | `intellihire/rag/retriever.py`, `prompts.py`, `guard.py`, `llm.py` |
| Agentic workflows (tool calling) | `intellihire/agent/tools.py`, `agent.py` |
| Streamlit dashboard + Plotly analytics | `app.py`, `ui/views/*`, `ui/charts.py` |
| Login, roles, audit | `intellihire/services/auth.py`, `users.py` |
| 100-candidate scenario, injection tests | `scripts/stress_test.py`, `tests/` |
| Streamlit Cloud / managed DB deployment | [§10](#10-deployment) |

## 2. Architecture

```mermaid
flowchart LR
    subgraph UI[Streamlit dashboard]
      J[Jobs] --- U[Upload] --- R[Rankings] --- C[Candidate] --- A[AI assistant] --- AN[Analytics] --- AD[Admin]
    end
    U -->|PDF/DOCX/TXT| V1[Validation]
    U -->|MP4/MOV/AVI| V2[Validation]
    V1 --> P[PyMuPDF / python-docx<br/>visible text + hidden-text quarantine]
    V2 --> O[OpenCV probe<br/>FFmpeg audio demux] --> W[Whisper STT]
    P --> N[spaCy skills, experience,<br/>education, certifications]
    W --> CM[Communication metrics]
    P --> E[MiniLM embeddings]
    W --> E
    N --> S[Scoring engine<br/>5 features, weighted]
    CM --> S
    E --> S
    S --> DB[(MySQL via SQLAlchemy)]
    E --> F[(FAISS index)]
    S --> ML[Scikit-learn / XGBoost<br/>learned ranker]
    DB --> RAG[Retriever + injection guard]
    F --> RAG
    RAG --> AG[Agent: tool calling] --> LLM[LLM API]
    AG --> A
    DB --> R & C & AN
```

## 3. Quick start (local)

Requires **Python 3.10–3.12** (3.11 recommended) and ~3 GB of disk for torch + models.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # CPU-only torch: pip install torch --index-url https://download.pytorch.org/whl/cpu  first
python -m spacy download en_core_web_sm
python scripts/download_models.py    # caches MiniLM + Whisper so the app never waits on the network

cp .env.example .env                 # optional: add an LLM key, MySQL URL, admin password
python scripts/generate_samples.py   # 2 job descriptions + 16 test resumes in sample_data/
python scripts/seed_demo.py          # loads them into the database (optional; re-running skips existing jobs)
streamlit run app.py
```

Open http://localhost:8501 and sign in with the bootstrap admin from `.env` (`ADMIN_USERNAME` / `ADMIN_PASSWORD`, default `admin` / `admin123`). Set a strong `ADMIN_PASSWORD` before the first start; if the default password is still in use, the app shows a warning and only the Admin page is available until you change it. After 5 failed sign-ins an account is locked for 15 minutes.

With no `DATABASE_URL`, IntelliHire uses a SQLite file in `data/` so it runs with zero setup. FFmpeg is bundled via `imageio-ffmpeg`; a system FFmpeg is used when present.

## 4. Using MySQL

```bash
docker compose up -d mysql
# in .env:
DATABASE_URL=mysql+pymysql://intellihire:intellihire@localhost:3306/intellihire
```

Tables are created automatically on first start (`init_db`). Any MySQL 8 / MariaDB 10.5+ server works; the connection uses `utf8mb4`. To run the whole stack in containers: `docker compose up --build` (app on port 8501).

## 5. Configuring the LLM

Set **one** of these in `.env`:

| Provider | Variables |
|---|---|
| Anthropic Claude | `ANTHROPIC_API_KEY=...` (optional `LLM_MODEL`, default `claude-sonnet-5-5`) |
| OpenAI | `OPENAI_API_KEY=...` (optional `LLM_MODEL`, default `gpt-4o-mini`) |
| Any OpenAI-compatible server (Groq, Together, local Ollama, LM Studio) | `LLM_PROVIDER=openai`, `OPENAI_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=llama3.1` |

Without a key the assistant runs in **offline mode**: a deterministic intent router calls the same tools (rankings, skill gaps, comparisons, semantic evidence search, interview-question plans) and renders their results as tables. Summaries and interview questions are template-based in this mode.

Speech-to-text defaults to local Whisper `base` on CPU. Set `WHISPER_MODE=api` to use the OpenAI Whisper API instead (needs `OPENAI_API_KEY`).

## 6. Using the dashboard

1. **Jobs** – paste or upload a JD → *Analyse*. Review the detected required/preferred skills, experience range and responsibilities, add custom skills, adjust the scoring weights, *Save*. Edit any job later and *Save & re-score*.
2. **Upload candidates** – drop up to 100 resumes plus optional videos. Videos pair with resumes by file name (`jane_doe.pdf` ↔ `jane_doe.mp4`). A results table reports added / duplicate / rejected files with reasons and security flags. Videos can also be attached later.
3. **Rankings** – top-5 cards and the full table with every sub-score. Filter by score, status, required skills, video, or flagged resumes. Edit the *Status* column to shortlist/reject, export CSV, open a profile. Once the learned ranker is trained you can rank by *ML* or a *Blend*.
4. **Candidate profile** – score gauge, waterfall of how each feature contributed, radar vs. the top-5 average, matched/missing skills **with the resume lines they came from**, experience evidence, video transcript with timestamps and communication metrics, AI summary and tailored interview questions with cited evidence, security findings, notes and status.
5. **AI assistant** – ask in plain language: *"Compare Priya and Rahul"*, *"Who has Kubernetes and 3+ years?"*, *"Skill gaps of #2"*, *"Interview questions for Neha"*. Every answer shows the agent's tool calls.
6. **Analytics** – score distribution, sub-score spread, missing-skill trends across the applicant pool, skill coverage, experience vs. score, pipeline status, processing timeline, and the learned-ranker panel (train, CV ROC-AUC per model, feature importance).
7. **Admin** – component health (shows any fallback in use), users and roles, vector-index rebuild / re-embed, audit log.

Toggle **Blind screening** in the sidebar to hide names and contact details everywhere, including what is sent to the LLM.

## 7. How scoring works

```
FINAL = Σ wᵢ · Sᵢ / Σ wᵢ     (over the features available for that candidate, × 100)
```

| Feature | Computation | Default weight |
|---|---|---|
| S_SEMANTIC | cosine(resume vector, JD vector), calibrated from MiniLM's typical range (0.15–0.70) to 0–1 | 0.30 |
| S_SKILL | (1.0 × required matched + 0.5 × preferred matched) / max; skills mentioned only in the video earn half credit | 0.35 |
| S_EXPERIENCE | years from merged employment date ranges (or explicit "N years"), compared with the JD range; below minimum scales linearly, far above maximum gets 0.85 | 0.20 |
| S_VIDEO | cosine(transcript vector, JD responsibilities vector), calibrated | 0.10 |
| S_COMM | 0.4 pace (120–165 wpm ideal) + 0.4 vocabulary diversity (MATTR) + 0.2 filler-word rate | 0.05 |

A candidate without a video is **not penalised**: video and communication weights are redistributed, and the explanation says so. Weights are editable per job.

The **learned ranker** treats shortlisted/interview/hired as positive and rejected as negative, cross-validates Logistic Regression, Random Forest and XGBoost on the sub-scores, keeps the best by ROC-AUC (needs ≥ 6 decisions with ≥ 2 per class) and shows its feature importances. It complements the formula; it never replaces the explanation.

## 8. Security, privacy and ethics

- **Validation before parsing** – type allow-list, size caps, file-signature checks, zip-bomb guard, sanitised filenames, max 100 files per batch.
- **Hidden-text quarantine** – white, microscopic, transparent or off-page text in PDFs/DOCX is separated and never scored, so keyword stuffing doesn't work.
- **Prompt-injection guard** – instruction-like sentences ("ignore previous instructions", "rank me first", "note to the AI screener") are flagged on the candidate, stripped before any text reaches an LLM, and candidate text is wrapped in `<candidate_data>` tags declared as untrusted. Scores are computed without an LLM, so injected text cannot change a ranking.
- **Agent sandbox** – agent tools are read-only and scoped to one job, with a bounded step count.
- **Data isolation** – relational data in MySQL; FAISS stores only vectors keyed by chunk id.
- **No SQL injection** – every query goes through the SQLAlchemy ORM with bound parameters.
- **Bias reduction** – video frames are never analysed (speech only); blind mode; the LLM is told not to consider protected characteristics; contact details are redacted before LLM calls.
- **Accounts** – bcrypt password hashes, admin/recruiter roles, login throttling, full audit log of logins, uploads, status changes and deletions.

Login throttling counts failed sign-ins per username+IP (`LOGIN_MAX_FAILURES`, default 5) and per IP across all usernames (`LOGIN_MAX_FAILURES_PER_IP`, default 20; 0 turns it off) within `LOGIN_WINDOW_MINUTES` (default 15). Behind a reverse proxy or Docker NAT every browser can appear to come from one address, so 20 bad passwords from anyone would block sign-in for everyone. If the app sits behind N trusted proxies, set `TRUSTED_PROXY_HOPS=N` so the client IP is read from `X-Forwarded-For` (the entry N places from the right); leave it at 0 (the default, header ignored) when clients can reach the app directly, because the header is then client-controlled. Without a proxy that sets the header, raise or disable `LOGIN_MAX_FAILURES_PER_IP` instead.

## 9. Testing

```bash
pytest                                 # unit, integration, LLM-adapter and headless UI tests
python scripts/stress_test.py          # 100-candidate scenario (pure pipeline)
python scripts/stress_test.py --with-db  # same, through the database
```

The suite covers validation, extraction (incl. hidden text and two-column PDFs), skill/experience parsing, scoring maths, the learned ranker, FFmpeg/OpenCV media handling, chunking, the injection guard, FAISS/NumPy vector stores, the full database pipeline, the offline agent, both LLM tool-calling adapters (with fake SDK clients) and every Streamlit page via `AppTest`.

## 10. Deployment

**Docker** – `docker compose up --build` runs MySQL + the app; models are pre-downloaded in the image. Set `MYSQL_PASSWORD` and `MYSQL_ROOT_PASSWORD` in `.env` before use. MySQL is not published to the host by default.

**Streamlit Community Cloud**
1. Push the repo to GitHub (the `.gitignore` excludes `data/`, `models/` and `.env`).
2. Create a managed MySQL database (e.g. Aiven, PlanetScale-compatible MySQL, AWS RDS).
3. In the app's *Settings → Secrets* add top-level keys (they become environment variables):
   ```toml
   DATABASE_URL = "mysql+pymysql://user:pass@host:3306/intellihire"
   ADMIN_PASSWORD = "a-strong-password"
   ANTHROPIC_API_KEY = "..."
   WHISPER_MODE = "api"          # recommended on Cloud: local Whisper is heavy for the free tier
   OPENAI_API_KEY = "..."
   ```
4. Set the main file to `app.py`. No system packages are needed: FFmpeg ships with `imageio-ffmpeg`.

The FAISS index lives on local disk; on ephemeral hosts it is rebuilt from the embeddings stored in MySQL (Admin → *Rebuild vector index*).

## 11. Project structure

```
intellihire/
├── app.py                     # Streamlit entry: auth gate + navigation
├── ui/
│   ├── common.py, charts.py   # shared widgets, Plotly figures
│   ├── pages.py               # page registry
│   └── views/                 # login, dashboard, jobs, upload, rankings, candidate, assistant, analytics, admin
├── intellihire/
│   ├── config.py              # env-driven settings
│   ├── ingestion/             # validation, PyMuPDF/python-docx extraction, normalisation
│   ├── nlp/                   # spaCy pipeline, skill taxonomy + matcher, experience, JD & resume parsers
│   ├── media/                 # OpenCV probe + FFmpeg audio, Whisper, communication metrics
│   ├── embeddings/            # MiniLM encoder
│   ├── scoring/               # 5-feature scoring, Scikit-learn/XGBoost ranker
│   ├── rag/                   # chunking, injection guard, FAISS/Chroma store, retriever, prompts, LLM clients
│   ├── agent/                 # tools + agent (LLM tool calling or offline router)
│   ├── db/                    # SQLAlchemy models + session
│   └── services/              # analysis (pure), pipeline (persisted), insights, users, auth, system status
├── scripts/                   # generate_samples, seed_demo, stress_test, download_models
├── tests/                     # pytest suite
├── sample_data/               # generated JDs and resumes
├── requirements.txt, .env.example, Dockerfile, docker-compose.yml
└── .streamlit/config.toml     # theme, 250 MB upload limit
```

## 12. Notes and limitations

- **OpenCV and audio.** OpenCV's `VideoCapture` only exposes the image stream, so IntelliHire uses OpenCV to validate the container and read metadata, and FFmpeg (the same dependency Whisper requires) to demux the audio track. Frames are never decoded.
- **Fallback modes.** If a model can't load (first run offline, missing package), the app keeps working and flags it on the dashboard and Admin page: hashing embeddings instead of MiniLM, blank spaCy tokenizer instead of `en_core_web_sm`, NumPy store instead of FAISS, pypdf instead of PyMuPDF (no hidden-text detection), offline agent instead of an LLM. Set `ALLOW_FALLBACKS=false` in production. After installing the real model, use Admin → *Re-embed all documents*.
- **Whisper on CPU** transcribes a 2-minute clip in roughly 20–60 s with the `base` model; use `WHISPER_MODEL=tiny` or `WHISPER_MODE=api` for speed.
- **Scanned PDFs** (images only) have no text layer; OCR is out of scope, and the upload result warns about them.
- Skill extraction is taxonomy-based for transparency. Add domain-specific skills per job as custom skills.
