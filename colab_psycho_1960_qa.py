# =============================================================================
# PSYCHO (1960) — ASK, GRADE, SAVE  |  full Colab notebook
# Runtime > Change runtime type > T4 GPU
# =============================================================================
#
# This is the FACTUAL 1960-film pipeline (not the transformed experiment).
# It writes to the Supabase table `psycho_qa_1960`.
#
# Before running:
#   1. Add two Colab secrets (key icon, left sidebar), Notebook access ON:
#        SUPABASE_URL   https://<your-ref>.supabase.co
#        SUPABASE_KEY   <legacy service_role JWT — starts eyJhbG..., ~200+ chars>
#      NOTE: use the LEGACY service_role key (Settings > API Keys > Legacy tab),
#      not the new sb_secret_... key — the pinned supabase client rejects the
#      new format.
#
#   2. Accept the licence for mistralai/Mistral-7B-Instruct-v0.3 on Hugging Face
#      once while logged in (the tokenizer config is pulled from it).
#
#   3. Create the table (Supabase SQL editor):
#
#        create table if not exists psycho_qa_1960 (
#            id          bigserial primary key,
#            date_asked  timestamptz not null default now(),
#            question    text not null,
#            answer      text not null,
#            grade       int check (grade between 1 and 10),
#            retrieved   text,
#            notes       text,
#            rag_on      boolean,
#            temperature real,
#            max_tokens  int
#        );
#
# The notebook can be run as one cell, or split at the numbered section
# markers. If you split it, run every section top-to-bottom in order;
# section 9 (the UI) must run last.
# =============================================================================


# ----------------------------------------------------------------- 1  install
!pip install -q \
  transformers==4.57.1 \
  accelerate==1.10.1 \
  bitsandbytes==0.48.1 \
  safetensors==0.6.2 \
  gradio==5.49.1 \
  sentence-transformers==5.1.1 \
  datasets==4.3.0 \
  faiss-cpu==1.12.0 \
  huggingface_hub==0.35.3 \
  supabase==2.9.1


# ----------------------------------------------------------------- 2  imports
import os, re, torch, faiss, gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from supabase import create_client
from google.colab import userdata

MODEL_NAME = "antfr99/mistral-7B-hitchcock-psycho-1960-film"
DATASET_NAME = "antfr99/psycho-1960-film-dataset"
TABLE = "psycho_qa_1960"
GRADE_MAX = 10                 # keep in sync with the Streamlit app
RELEVANCE_THRESHOLD = 1.0      # off-topic gate; tune against your own measurements


# ----------------------------------------------------------------- 3  supabase
sb = create_client(userdata.get("SUPABASE_URL"), userdata.get("SUPABASE_KEY"))
print("Supabase client ready")


# ----------------------------------------------------------------- 4  model
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=bnb, device_map="auto"
)
model.eval()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Model loaded")


# ----------------------------------------------------------------- 5  retrieval
dataset = load_dataset(DATASET_NAME, split="train")
passages = [f"{r['prompt']} {r['completion']}" for r in dataset]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=True)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)
print(f"FAISS index built over {len(passages)} passages")


# ----------------------------------------------------------------- 6  keywords
# Loaded for reference only. The list is a vocabulary dump of the dataset and
# passes almost any English sentence, so it is NOT the off-topic filter — the
# retrieval-distance gate in section 7 does that job.
STOPWORDS = {
    "the","in","her","his","their","a","an","and","or","of","on","for","to","with",
    "at","by","is","was","as","from","that","which","it","he","she","they","this",
    "these","those","but","not","had","have","has","also","after","before","so",
    "would","could","should","who","what","when","where","why","how","does","did",
    "do","i","you","we","me","my","your",
}

try:
    kw_dir = snapshot_download(repo_id=MODEL_NAME, allow_patterns=["psycho_keywords.txt"])
    with open(os.path.join(kw_dir, "psycho_keywords.txt"), encoding="utf-8") as f:
        KEYWORDS = {ln.strip().lower().rstrip("s") for ln in f if ln.strip()}
    print(f"Loaded {len(KEYWORDS)} keywords (reference only)")
except Exception as e:
    KEYWORDS = set()
    print(f"Keyword file not loaded ({e}) — continuing without it")


# ----------------------------------------------------------------- 7  generate
# Returns SIX values: answer, retrieved, note, and the three settings used
# (so save_row can store exactly what produced this answer).
def ask_model(question, max_new_tokens, temperature, grounded):
    question = (question or "").strip()
    if not question:
        return "", "", "Enter a question first.", grounded, float(temperature), int(max_new_tokens)

    print(f"[DEBUG] tokens={max_new_tokens}  temp={temperature}  grounded={grounded}")

    # --- off-topic gate via retrieval distance (runs regardless of grounding)
    qv = embedder.encode([question], convert_to_numpy=True)
    D, idx = index.search(qv, k=3)
    nearest = float(D[0][0])

    if nearest > RELEVANCE_THRESHOLD:
        print(f"[DEBUG] refused as off-topic, distance={nearest:.3f}")
        return (
            "This question doesn't appear to be about Psycho (1960), so I won't answer it.",
            "",
            f"Off-topic (nearest passage distance {nearest:.2f} > {RELEVANCE_THRESHOLD})",
            grounded, float(temperature), int(max_new_tokens),
        )

    # --- build the prompt
    if grounded:
        retrieved = "\n\n---\n\n".join(passages[i] for i in idx[0])
        # Softened instruction: passages are the PRIMARY source, but the model
        # may fill gaps with accurate on-topic facts. Strict "only these
        # passages" wording caused failures when a fact (e.g. the release year)
        # wasn't in the retrieved chunk.
        content = (
            "You are a film scholar answering about the 1960 film 'Psycho' by "
            "Alfred Hitchcock. Use the passages below as your primary source. "
            "If they don't fully cover the question, you may add accurate, "
            "directly relevant facts about the film. Do not invent anything or "
            "go off-topic.\n\n"
            f"{retrieved}\n\nQuestion: {question}"
        )
        print(f"[DEBUG] grounded=True — {len(retrieved)} chars retrieved")
    else:
        retrieved = ""
        content = question
        print("[DEBUG] grounded=False — no retrieved context")

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)

    do_sample = temperature > 0
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=max(temperature, 1e-4),
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )
    answer = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    print(f"[DEBUG] generated {len(out[0]) - ids.shape[-1]} tokens")

    return (
        answer,
        retrieved,
        f"On-topic (distance {nearest:.2f}, grounded={grounded})",
        grounded, float(temperature), int(max_new_tokens),
    )


# ----------------------------------------------------------------- 8  save
def save_row(question, answer, grade, notes, retrieved,
             rag_on, temperature, max_tokens):
    if not (question or "").strip() or not (answer or "").strip():
        return "Nothing to save — ask a question first."
    try:
        res = (
            sb.table(TABLE)
            .insert(
                {
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "grade": int(grade),
                    "notes": (notes or "").strip() or None,
                    "retrieved": retrieved or None,
                    "rag_on": bool(rag_on),
                    "temperature": float(temperature),
                    "max_tokens": int(max_tokens),
                }
            )
            .execute()
        )
        row_id = res.data[0]["id"] if res.data else "?"
        return (
            f"Saved as row {row_id} "
            f"(grade {int(grade)}/{GRADE_MAX}, RAG={bool(rag_on)}, "
            f"temp={float(temperature):g}, tokens={int(max_tokens)})"
        )
    except Exception as e:
        return f"Save failed: {e}"


# ----------------------------------------------------------------- 9  UI
with gr.Blocks(title="Psycho (1960) — Ask & Grade") as demo:
    gr.Markdown(
        "## Psycho (1960) — Ask, Grade, Save\n"
        "Ask the fine-tuned model, grade the answer, then save it (with the "
        "settings used) to Supabase. Graded rows appear in the Streamlit archive."
    )

    # hidden state: the settings that produced the CURRENT answer, so a slider
    # nudged before saving can't change what gets recorded
    used_rag    = gr.State(True)
    used_temp   = gr.State(0.0)
    used_tokens = gr.State(200)

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="Question", lines=4,
                placeholder="e.g. Who composed the score for Psycho?",
            )
            with gr.Row():
                grounded = gr.Checkbox(label="Ground with retrieval (RAG)", value=True)
                max_tokens = gr.Slider(64, 512, value=200, step=32, label="Max new tokens")
            temperature = gr.Slider(
                0.0, 1.0, value=0.0, step=0.05,
                label="Temperature (0 = deterministic, best for factual checks)",
            )
            ask_btn = gr.Button("Ask the model", variant="primary")

        with gr.Column(scale=1):
            answer = gr.Textbox(label="Answer", lines=10, show_copy_button=True)
            note = gr.Markdown()

    with gr.Accordion("Passages used for grounding", open=False):
        retrieved = gr.Textbox(label="", lines=8, show_label=False)

    gr.Markdown("### Grade this answer")
    with gr.Row():
        grade = gr.Slider(
            1, GRADE_MAX, value=GRADE_MAX // 2, step=1,
            label=f"Factual accuracy (1–{GRADE_MAX})",
        )
        notes = gr.Textbox(label="Grading notes (optional)", lines=2,
                           placeholder="e.g. correct on Herrmann, wrong on the budget")
    save_btn = gr.Button("Save to Supabase", variant="primary")
    status = gr.Markdown()

    ask_btn.click(
        ask_model,
        inputs=[question, max_tokens, temperature, grounded],
        outputs=[answer, retrieved, note, used_rag, used_temp, used_tokens],
    )
    save_btn.click(
        save_row,
        inputs=[question, answer, grade, notes, retrieved,
                used_rag, used_temp, used_tokens],
        outputs=[status],
    )

demo.launch(share=True, debug=True)
