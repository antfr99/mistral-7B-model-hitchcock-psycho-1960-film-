# =============================================================================
# PSYCHO (1960) — ASK, GRADE, SAVE  |  single Colab cell
# Runtime > Change runtime type > T4 GPU
# =============================================================================
#
# Before running, add two Colab secrets (🔑 icon in the left sidebar):
#     SUPABASE_URL   https://<your-ref>.supabase.co
#     SUPABASE_KEY   <service_role key — needs insert permission>
#
# And create the table in Supabase (SQL editor):
#
#     create table psycho_qa_1960 (
#         id          bigserial primary key,
#         date_asked  timestamptz not null default now(),
#         question    text not null,
#         answer      text not null,
#         grade       int,
#         retrieved   text,
#         notes       text
#     );
#
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
GRADE_MAX = 10          # keep in sync with the Streamlit app

# ----------------------------------------------------------------- 3  supabase
sb = create_client(userdata.get("SUPABASE_URL"), userdata.get("SUPABASE_KEY"))
print("✅ Supabase client ready")

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
print("✅ Model loaded")

# ----------------------------------------------------------------- 5  retrieval
dataset = load_dataset(DATASET_NAME, split="train")
passages = [f"{r['prompt']} {r['completion']}" for r in dataset]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
emb = embedder.encode(passages, convert_to_numpy=True, show_progress_bar=True)
index = faiss.IndexFlatL2(emb.shape[1])
index.add(emb)
print(f"✅ FAISS index built over {len(passages)} passages")

# ----------------------------------------------------------------- 6  keywords
# Weak off-topic filter only. The list is a vocabulary dump of the dataset and
# passes almost any English sentence, so it is NOT an anti-hallucination guard —
# the retrieval in section 7 is what actually constrains the answer.
STOPWORDS = {
    "the","in","her","his","their","a","an","and","or","of","on","for","to","with",
    "at","by","is","was","as","from","that","which","it","he","she","they","this",
    "these","those","but","not","had","have","has","also","after","before","so",
    "would","could","should","who","what","when","where","why","how","does","did",
    "do","i","you","we","me","my","your",
}

kw_dir = snapshot_download(repo_id=MODEL_NAME, allow_patterns=["psycho_keywords.txt"])
kw_file = os.path.join(kw_dir, "psycho_keywords.txt")
with open(kw_file, encoding="utf-8") as f:
    KEYWORDS = {ln.strip().lower().rstrip("s") for ln in f if ln.strip()}
print(f"✅ Loaded {len(KEYWORDS)} keywords")


def relevance_note(prompt):
    words = re.findall(r"\b[A-Za-z0-9]+\b", re.sub(r"[’']", "", prompt).lower())
    off = [w for w in words if w.rstrip("s") not in KEYWORDS and w not in STOPWORDS]
    return ("⚠️ Possibly off-topic terms: " + ", ".join(off)) if off else "✅ On-topic"


# ----------------------------------------------------------------- 7  generate
def ask_model(question, max_new_tokens, temperature, grounded):
    question = (question or "").strip()
    if not question:
        return "", "", "Enter a question first."

    note = relevance_note(question)

    retrieved = ""
    if grounded:
        qv = embedder.encode([question], convert_to_numpy=True)
        _, idx = index.search(qv, k=3)
        retrieved = "\n\n---\n\n".join(passages[i] for i in idx[0])
        content = (
            "You are a film scholar. Answer strictly about the 1960 film 'Psycho' "
            "by Alfred Hitchcock. Do not invent details. Base your answer only on "
            "the passages below.\n\n"
            f"{retrieved}\n\nQuestion: {question}"
        )
    else:
        content = question

    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-4),
            top_p=0.95,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
        )

    answer = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return answer, retrieved, note


# ----------------------------------------------------------------- 8  save
def save_row(question, answer, grade, notes, retrieved):
    if not (question or "").strip() or not (answer or "").strip():
        return "❌ Nothing to save — ask a question first."
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
                }
            )
            .execute()
        )
        row_id = res.data[0]["id"] if res.data else "?"
        return f"✅ Saved as row {row_id} (grade {int(grade)}/{GRADE_MAX})"
    except Exception as e:
        return f"❌ Save failed: {e}"


# ----------------------------------------------------------------- 9  UI
with gr.Blocks(title="Psycho (1960) — Ask & Grade") as demo:
    gr.Markdown(
        "## 🎬 Psycho (1960) — Ask, Grade, Save\n"
        "Ask the fine-tuned model, grade the answer for factual accuracy, "
        "then save it to Supabase. Graded rows appear in the Streamlit archive."
    )

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
    save_btn = gr.Button("💾 Save to Supabase", variant="primary")
    status = gr.Markdown()

    ask_btn.click(
        ask_model,
        inputs=[question, max_tokens, temperature, grounded],
        outputs=[answer, retrieved, note],
    )
    save_btn.click(
        save_row,
        inputs=[question, answer, grade, notes, retrieved],
        outputs=[status],
    )

demo.launch(share=True, debug=True)
