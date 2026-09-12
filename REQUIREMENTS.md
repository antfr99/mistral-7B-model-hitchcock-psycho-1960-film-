# Psycho (1960) Q&A Pipeline — Requirements

## Overview

A three-part pipeline for running factual questions against a fine-tuned
Mistral-7B model themed on Alfred Hitchcock's *Psycho (1960)*, grading the
answers for accuracy, persisting results to Supabase, and displaying the
archive in a read-only Streamlit app.

The pipeline is entirely separate from the transformed-model experiment
(`psycho_qa` table / Psycho 2026 app). Nothing from the fiction experiment
appears in this archive, and vice versa.

---

## Components

| Component | File | Purpose |
|---|---|---|
| Colab notebook | `colab_psycho_1960_qa.py` | Load model, ask questions, grade, save to Supabase |
| Streamlit viewer | `app.py` | Read-only display of the Q&A archive |
| Supabase table | see SQL below | Persistent storage |

---

## 1. Google Colab (inference + grading)

### Runtime requirement
- **GPU: T4** (free tier sufficient)
- Runtime → Change runtime type → T4 GPU

### Python libraries (pinned — do not upgrade without testing)

| Library | Version | Purpose |
|---|---|---|
| `transformers` | 4.57.1 | Model loading, tokenizer, generation |
| `accelerate` | 1.10.1 | `device_map="auto"` multi-device dispatch |
| `bitsandbytes` | 0.48.1 | 4-bit NF4 quantisation |
| `safetensors` | 0.6.2 | Efficient weight loading |
| `gradio` | 5.49.1 | In-Colab UI (ask + grade in one cell) |
| `sentence-transformers` | 5.1.1 | Passage embedding for FAISS retrieval |
| `datasets` | 4.3.0 | Load `antfr99/psycho-1960-film-dataset` from HF |
| `faiss-cpu` | 1.12.0 | Nearest-neighbour retrieval over passages |
| `huggingface_hub` | 0.35.3 | `snapshot_download` for keyword file |
| `supabase` | 2.9.1 | Python client for Supabase insert |

> **Why pinned?** The original inference code used `pip install --upgrade` with no
> version pins. When Colab updated its library cache the code broke silently —
> the model ran but produced hallucinated output due to changed BOS-token and
> chat-template behaviour in newer `transformers`. These versions match the
> October 2025 environment in which the model was known to work correctly.

### Colab secrets (🔑 left sidebar → Secrets)

| Secret name | Value |
|---|---|
| `SUPABASE_URL` | `https://<your-ref>.supabase.co` |
| `SUPABASE_KEY` | Service-role key (needs INSERT on `psycho_qa_1960`) |

### Hugging Face access

The tokenizer config is pulled from `mistralai/Mistral-7B-Instruct-v0.3`.
That repo is **gated** — accept its licence on Hugging Face once while logged
in before running the notebook, otherwise the tokenizer download will fail with
a 403.

The model weights (`antfr99/mistral-7B-hitchcock-psycho-1960-film`) are public
and require no special access.

---

## 2. Streamlit app (viewer)

### Python libraries

| Library | Version | Purpose |
|---|---|---|
| `streamlit` | latest stable | Web app framework |
| `supabase` | 2.9.1 | Read from Supabase |
| `pandas` | latest stable | DataFrame handling |

Pin `supabase` to match the Colab install so the client behaviour is identical.
`streamlit` and `pandas` are safe to leave unpinned for the viewer.

### `requirements.txt` for Streamlit Cloud

```
streamlit
pandas
supabase==2.9.1
```

### Streamlit Cloud secrets (Settings → Secrets)

```toml
SUPABASE_URL = "https://<your-ref>.supabase.co"
SUPABASE_KEY = "<anon key — read-only is sufficient for the viewer>"
```

Use the **anon key** in the Streamlit app (read-only). Use the **service-role
key** in Colab (needs INSERT). Never put the service-role key in a public
Streamlit app.

---

## 3. Supabase

### Table: `psycho_qa_1960`

See `create_table_psycho_qa_1960.sql` for the full creation script.

| Column | Type | Notes |
|---|---|---|
| `id` | `bigserial` PK | Auto-incrementing row identifier |
| `date_asked` | `timestamptz` | Defaults to `now()`, set server-side |
| `question` | `text NOT NULL` | The question sent to the model |
| `answer` | `text NOT NULL` | The raw model output |
| `grade` | `int` (nullable) | 1–10 factual accuracy score |
| `retrieved` | `text` (nullable) | The three FAISS passages injected into the prompt |
| `notes` | `text` (nullable) | Free-text grading comment |

### Row-level security

The table uses RLS. The anon key can only SELECT (for the Streamlit viewer).
The service-role key used in Colab bypasses RLS entirely, so no explicit INSERT
policy is needed for that key — but enabling RLS is still recommended so the
anon key cannot write.

---

## Data flow

```
Google Colab (T4 GPU)
│
├── Load model:  antfr99/mistral-7B-hitchcock-psycho-1960-film
├── Load dataset: antfr99/psycho-1960-film-dataset → FAISS index
├── Load keywords: psycho_keywords.txt (off-topic advisory only)
│
├── User types question in Gradio UI
├── FAISS retrieves top-3 passages → injected into prompt
├── Model generates answer via apply_chat_template()
├── User grades 1–10 + optional notes
└── INSERT → Supabase (psycho_qa_1960)
                │
                └── Streamlit app polls every 30s
                    └── Displays Q&A cards with grade colour coding
```

---

## Key technical decisions

**`apply_chat_template()` not `<s>[INST]...[/INST]`**
Current `transformers` adds the BOS token automatically. The old hand-built
prompt string prepended it again, causing a double-BOS that degraded or emptied
output. `apply_chat_template()` is the correct way to format prompts and is
stable across versions.

**Merged model, no PeftModel**
The HF repo is a full merged model served as Safetensors. It loads with
`AutoModelForCausalLM.from_pretrained()` directly. No `PeftModel` adapter step
is needed (that is the transformed-model repo, `psycho-mistral-v03-transformed-adapter`).

**Temperature = 0 default**
For factual grading, deterministic output is preferable so the same question
gives the same answer across sessions. Temperature can be raised for exploratory
or creative queries.

**Separate table from the transformed model**
The transformed model experiment writes to `psycho_qa`. This pipeline writes to
`psycho_qa_1960`. The two must never share a table — one set of answers is
factual (graded for accuracy), the other is deliberately fictional (graded for
staying in-world).

**`retrieved` column**
Storing the passages used for grounding makes low grades diagnosable: you can
see whether the retrieval was poor (wrong passages) or the model ignored correct
ones. Without this, a low grade is just a number.
