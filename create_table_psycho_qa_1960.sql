-- =============================================================================
-- Psycho (1960) Q&A — Supabase table setup
-- Run this in the Supabase SQL Editor (project → SQL Editor → New query)
-- =============================================================================
-- This is the FACTUAL model archive. Keep it separate from the transformed-
-- model experiment which uses the table "psycho_qa" (Psycho 2026 app).
-- =============================================================================


-- -----------------------------------------------------------------------------
-- 1. Create the table
-- -----------------------------------------------------------------------------

create table if not exists psycho_qa_1960 (

    id          bigserial       primary key,

    -- When the question was asked. Set server-side so there is no clock-skew
    -- between Colab and Supabase.
    date_asked  timestamptz     not null default now(),

    -- The question sent to the model. Cannot be empty.
    question    text            not null,

    -- The raw model output. Cannot be empty.
    answer      text            not null,

    -- Factual accuracy score 1–10 (nullable until graded).
    -- Adjust the check constraint if you prefer a 1–5 scale.
    grade       int             check (grade between 1 and 10),

    -- The three FAISS-retrieved passages that were injected into the prompt.
    -- Nullable — only populated when the RAG / grounding flag was on.
    -- Stored so a low grade can be traced: bad retrieval vs model ignoring
    -- good passages.
    retrieved   text,

    -- Free-text grading note, e.g. "correct on Herrmann, wrong on budget".
    notes       text

);


-- -----------------------------------------------------------------------------
-- 2. Index on date_asked so the Streamlit ORDER BY query stays fast as the
--    table grows.
-- -----------------------------------------------------------------------------

create index if not exists psycho_qa_1960_date_idx
    on psycho_qa_1960 (date_asked desc);


-- -----------------------------------------------------------------------------
-- 3. Enable Row-Level Security
--    This keeps the anon key (used in the public Streamlit app) read-only.
--    The service-role key used in Colab bypasses RLS automatically, so no
--    INSERT policy is needed for that key.
-- -----------------------------------------------------------------------------

alter table psycho_qa_1960 enable row level security;


-- Allow anyone with the anon key to SELECT (read-only Streamlit viewer).
create policy "Public read"
    on psycho_qa_1960
    for select
    using (true);

-- No INSERT / UPDATE / DELETE policy for the anon key — only the service-role
-- key (used in Colab) can write to the table.


-- -----------------------------------------------------------------------------
-- 4. Verify — should return 0 rows on a fresh table
-- -----------------------------------------------------------------------------

select count(*) as row_count from psycho_qa_1960;


-- =============================================================================
-- Optional: seed one test row to confirm the pipeline end-to-end before
-- running the model. Delete it once the real data starts flowing.
-- =============================================================================

/*
insert into psycho_qa_1960 (question, answer, grade, notes)
values (
    'Who directed Psycho?',
    'Alfred Hitchcock directed Psycho. The film was released in 1960 and produced '
    'by Hitchcock himself under his Shamley Productions company.',
    9,
    'Test row — delete once real data is flowing.'
);
*/
