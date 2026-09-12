"""
Psycho (1960) Q&A Viewer — read-only Streamlit app over the Supabase `psycho_qa_1960` table.

This is the FACTUAL model archive (Hitchcock's real 1960 film), not the
transformed / AI-reimagined experiment. Keep it in its own table so the two
never mix.

Supabase schema:

    create table psycho_qa_1960 (
        id          bigserial primary key,
        date_asked  timestamptz not null default now(),
        question    text not null,
        answer      text not null,
        grade       int,              -- 1..GRADE_MAX, nullable
        retrieved   text,             -- optional: the RAG passages used
        notes       text              -- optional: grader's comment
    );

Secrets required (Streamlit Cloud: Settings > Secrets, or .streamlit/secrets.toml):
    SUPABASE_URL = "https://<your-ref>.supabase.co"
    SUPABASE_KEY = "<anon or service_role key>"
"""

import streamlit as st
import pandas as pd
from supabase import create_client

# ------------------------------------------------------------------ config
st.set_page_config(page_title="Psycho (1960) Q&A Viewer", page_icon="🎬", layout="wide")

TABLE = "psycho_qa_1960"
GRADE_MAX = 10          # set to 5 if you prefer a 1-5 scale
MODEL_URL = "https://huggingface.co/antfr99/mistral-7B-hitchcock-psycho-1960-film"
DATASET_URL = "https://huggingface.co/datasets/antfr99/psycho-1960-film-dataset"


# ------------------------------------------------------------------ data
@st.cache_resource
def get_client():
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


@st.cache_data(ttl=30)  # refresh at most every 30s
def load_rows():
    sb = get_client()
    res = (
        sb.table(TABLE)
        .select(
            "id, question, answer, grade, retrieved, notes, date_asked, "
            "rag_on, temperature, max_tokens"
        )
        .order("date_asked", desc=True)
        .execute()
    )
    df = pd.DataFrame(res.data or [])
    if not df.empty and "date_asked" in df:
        df["date_asked"] = pd.to_datetime(df["date_asked"])
    return df


# ------------------------------------------------------------------ style
# Monochrome palette — deliberately different from the transformed-model app
# so the two archives are never confused at a glance.
st.markdown(
    """
    <style>
      .stApp { background:#ffffff; }
      .block-container { padding-top:2.2rem; max-width:1100px; }
      h1, h2, h3, p, label, span, div { color:#1a1a1a; }
      a, a:visited { color:#2b2b2b; text-decoration:underline; }
      a:hover { color:#000000; }
      .qa-card {
        border-left:3px solid #2b2b2b; background:#f4f4f4;
        padding:1.1rem 1.3rem; margin-bottom:1rem; border-radius:2px;
      }
      .qa-q { font-size:1.05rem; font-weight:600; color:#1a1a1a; margin-bottom:.5rem; }
      .qa-a { color:#3a3a3a; line-height:1.55; }
      .qa-meta { color:#8a857d; font-size:.8rem; margin-top:.7rem;
                 letter-spacing:.02em; }
      .qa-notes { color:#5a5a5a; font-size:.85rem; font-style:italic; margin-top:.5rem; }
      .grade-pill { display:inline-block; padding:.1rem .55rem; border-radius:999px;
                    font-weight:700; font-size:.8rem; }
      .qa-settings { margin-top:.6rem; display:flex; gap:.4rem; flex-wrap:wrap; }
      .set-pill { display:inline-block; padding:.1rem .5rem; border-radius:4px;
                  font-size:.72rem; font-weight:600; background:#e8e6e2; color:#4a4a4a;
                  letter-spacing:.02em; }
      .set-on  { background:#dbe9f5; color:#1c4a7a; }
      .set-off { background:#f0e2d8; color:#8a4a1c; }
    </style>
    """,
    unsafe_allow_html=True,
)


def grade_color(g):
    """Proportional thresholds so this works for any GRADE_MAX."""
    if g is None or pd.isna(g):
        return "#e8e6e2", "#6a655d", "ungraded"
    g = int(g)
    pct = g / GRADE_MAX
    label = f"{g}/{GRADE_MAX}"
    if pct >= 0.8:
        return "#d6f0df", "#1c7a45", label
    if pct >= 0.5:
        return "#f5ecd0", "#8a6d1c", label
    return "#f5d9d9", "#a02c2c", label


# ------------------------------------------------------------------ header
st.title("🎬 Psycho (1960) — Q&A Archive")
st.caption(
    "A read-only record of questions put to the fine-tuned model, and how "
    "factually accurate each answer was judged to be."
)

st.markdown(
    f"🤗 **Model:** [{MODEL_URL.split('huggingface.co/')[-1]}]({MODEL_URL}) "
    f"&nbsp;·&nbsp; **Dataset:** [{DATASET_URL.split('huggingface.co/')[-1]}]({DATASET_URL})"
)

with st.expander("About this model", expanded=True):
    st.markdown(
        """
This archive records answers from a **Mistral-7B-Instruct-v0.3** model fine-tuned
(QLoRA) on material about Alfred Hitchcock's ***Psycho* (1960)** — the real film,
its production, cast, score, and reception.

Answers are **grounded by retrieval**: before generating, the question is embedded
and the three closest passages from the training dataset are pulled via FAISS and
placed in the prompt. The model is instructed to answer from those passages rather
than from its own general knowledge. Each answer is then graded by hand for
factual accuracy.

**This is the factual archive.** A separate experiment trained an adapter on a
deliberately rewritten version of the story; its answers are fiction by design and
live in their own app and table. Nothing from that experiment appears here.
        """
    )

with st.expander("How grading works"):
    st.markdown(
        f"""
Each answer is scored **1–{GRADE_MAX}** on how factually accurate it is about the
real 1960 film:

- **High** — accurate, specific, and supported by the retrieved passages.
- **Middle** — broadly correct but vague, padded, or partly unsupported.
- **Low** — invented details, wrong names or dates, or drifting off the film entirely.

Where the grading notes are filled in, they appear under the answer. The retrieved
passages used for grounding are stored alongside each row, so a low grade can be
traced back to whether the retrieval was poor or the model ignored good passages.
        """
    )

with st.expander("Test the model yourself"):
    st.markdown(
        """
This repo is a **merged model** — load it directly, no adapter step. Runs on a
free Google Colab **T4 GPU**.

**1. Install pinned libraries** (unpinned installs are what broke earlier runs):
        """
    )
    st.code(
        "pip install -q transformers==4.57.1 accelerate==1.10.1 "
        "bitsandbytes==0.48.1 sentence-transformers==5.1.1 faiss-cpu==1.12.0",
        language="bash",
    )
    st.markdown("**2. Load it and ask something:**")
    st.code(
        '''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "antfr99/mistral-7B-hitchcock-psycho-1960-film"

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto"
)
model.eval()

def ask(q, max_new_tokens=200):
    ids = tok.apply_chat_template(          # <-- do NOT hand-build "<s>[INST]..."
        [{"role": "user", "content": q}],
        add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                         repetition_penalty=1.2, no_repeat_ngram_size=3,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()

print(ask("Who composed the score for Psycho?"))
print(ask("How was the shower scene filmed?"))''',
        language="python",
    )
    st.markdown(
        """
Use `tokenizer.apply_chat_template()` rather than writing `<s>[INST] ... [/INST]`
by hand. Current `transformers` adds the BOS token itself, so a hand-written one
is added twice and the output degrades.
        """
    )

try:
    df = load_rows()
except Exception as e:
    st.error(f"Couldn't reach Supabase. Check your app secrets.\n\n{e}")
    st.stop()

if df.empty:
    st.info("No entries yet. Once you run questions through the model, they'll appear here.")
    st.stop()

# ------------------------------------------------------------------ stats
graded = df[df["grade"].notna()] if "grade" in df else pd.DataFrame()
c1, c2, c3 = st.columns(3)
c1.metric("Questions logged", len(df))
c2.metric("Graded", len(graded))
c3.metric(
    "Average grade",
    f"{graded['grade'].mean():.1f}/{GRADE_MAX}" if len(graded) else "—",
)

# ------------------------------------------------------------------ filters
st.divider()
f1, f2 = st.columns([2, 1])
with f1:
    search = st.text_input(
        "Search questions or answers",
        placeholder="e.g. Herrmann, shower scene, Arbogast, budget",
    )
with f2:
    only_graded = st.selectbox("Show", ["All", "Graded only", "Ungraded only"])

view = df.copy()
if search:
    s = search.lower()
    view = view[
        view["question"].str.lower().str.contains(s, na=False)
        | view["answer"].str.lower().str.contains(s, na=False)
    ]
if only_graded == "Graded only":
    view = view[view["grade"].notna()]
elif only_graded == "Ungraded only":
    view = view[view["grade"].isna()]

st.caption(f"Showing {len(view)} of {len(df)} entries")

# ------------------------------------------------------------------ list
for _, r in view.iterrows():
    bg, fg, label = grade_color(r.get("grade"))
    if pd.notna(r.get("date_asked")):
        iso = r["date_asked"].isocalendar()
        when = f"Week {iso.week}, {iso.year}"
    else:
        when = "unknown date"

    notes = r.get("notes")
    notes_html = (
        f'<div class="qa-notes">{notes}</div>'
        if isinstance(notes, str) and notes.strip()
        else ""
    )

    # --- settings pills (RAG / temperature / tokens) ---
    settings_bits = []
    rag_on = r.get("rag_on")
    if pd.notna(rag_on):
        if rag_on:
            settings_bits.append('<span class="set-pill set-on">RAG on</span>')
        else:
            settings_bits.append('<span class="set-pill set-off">RAG off</span>')
    temp = r.get("temperature")
    if pd.notna(temp):
        settings_bits.append(f'<span class="set-pill">temp {float(temp):g}</span>')
    toks = r.get("max_tokens")
    if pd.notna(toks):
        settings_bits.append(f'<span class="set-pill">{int(toks)} tok</span>')
    settings_html = (
        f'<div class="qa-settings">{"".join(settings_bits)}</div>'
        if settings_bits else ""
    )

    st.markdown(
        f"""
        <div class="qa-card">
          <div class="qa-q">{r['question']}</div>
          <div class="qa-a">{r['answer']}</div>
          {notes_html}
          {settings_html}
          <div class="qa-meta">
            {when} &nbsp;·&nbsp; row {r['id']} &nbsp;·&nbsp;
            <span class="grade-pill" style="background:{bg};color:{fg};">{label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    retrieved = r.get("retrieved")
    if isinstance(retrieved, str) and retrieved.strip():
        with st.expander(f"Passages retrieved for row {r['id']}"):
            st.text(retrieved)

# ------------------------------------------------------------------ table + download
with st.expander("View as table / download"):
    table = view.copy()
    if "retrieved" in table:
        table = table.drop(columns=["retrieved"])   # too long for a table view
    if "date_asked" in table:
        table["date_asked"] = table["date_asked"].dt.strftime("%B %Y")
        cols = ["date_asked"] + [c for c in table.columns if c != "date_asked"]
        table = table[cols]
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        table.to_csv(index=False).encode("utf-8"),
        file_name="psycho_qa_1960.csv",
        mime="text/csv",
    )