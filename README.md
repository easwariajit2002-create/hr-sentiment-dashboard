# HR Employee Sentiment Intelligence

A reusable, interactive HR sentiment dashboard, deployable as a public
Streamlit Community Cloud web app (`https://<something>.streamlit.app`).
The evaluator opens the URL, uploads a CSV/Excel export, and gets the full
dashboard — no Python, Colab, or local install required on their end.

## Deploying it (you'll need a free GitHub + Streamlit Community Cloud account)

I can't create accounts or push to GitHub on your behalf, but the repo is
100% ready to go — this is the whole process:

1. **Push this folder to a new GitHub repo.**
   ```bash
   cd hr_sentiment_app
   git init
   git add .
   git commit -m "HR sentiment intelligence dashboard"
   git branch -M main
   git remote add origin https://github.com/<you>/hr-sentiment-app.git
   git push -u origin main
   ```
   (The repo can be public or private — Streamlit Community Cloud supports
   both once your GitHub account is connected.)

2. **Go to [share.streamlit.io](https://share.streamlit.io)**, sign in with
   GitHub, click **"New app"**, and select:
   - Repository: `<you>/hr-sentiment-app`
   - Branch: `main`
   - Main file path: `app.py`

3. Click **Deploy**. The first build installs `requirements.txt` (a couple
   of minutes, since it includes CPU-only PyTorch). Once it's live you get a
   URL like `https://hr-sentiment-app-<hash>.streamlit.app` — that's what you
   submit.

4. The **first visitor** after each deploy/restart will wait ~30-60 seconds
   while the ~500MB RoBERTa model downloads from Hugging Face. Every
   visitor after that hits the in-memory cached model (see "Model caching"
   below) — no repeat downloads for the life of that running instance.

**Free-tier resource note:** Streamlit Community Cloud's free tier has a
memory ceiling (historically ~1GB). `torch` + `transformers` +
`roberta-base` fit, but it's not a lot of headroom alongside pandas on a
large upload. If you hit out-of-memory errors on very large files, the
easiest fixes are: (a) ask Streamlit Community Cloud support for a resource
bump on your account, or (b) redeploy the same repo unchanged on
[Hugging Face Spaces](https://huggingface.co/spaces) with the "Streamlit"
SDK — same `app.py`, same `requirements.txt`, more generous free RAM, and a
public URL of its own (`https://<you>-<space>.hf.space`) if `.streamlit.app`
turns out not to work for your evaluator's file sizes.

## Model caching — what's actually guaranteed

- `@st.cache_resource` wraps model loading in `app.py`. This means the
  RoBERTa model is loaded into memory **once per running app instance**,
  and every user session after the first reuses that same in-memory model —
  it is never reloaded per session, per upload, or per filter change.
- `HF_HOME` is pinned to a folder inside the app directory so the downloaded
  weights sit in one predictable place on disk for the life of the
  container (matters if the instance restarts without a full redeploy —
  e.g. waking from Streamlit's "app went to sleep" state, which usually
  preserves disk).
- What is **not** guaranteed on the free tier: a full redeploy (new commit,
  or Streamlit rebuilding the container from scratch) gets a fresh
  filesystem, so that one instance will re-download the model once. This is
  a hosting-platform constraint, not something `st.cache_resource` can work
  around — there's no persistent-volume option on the free tier to point
  the download at instead.

## What's in the repo

```
app.py                          Streamlit entrypoint — this is what you deploy
.streamlit/config.toml          Upload size limit, theme (read automatically by Cloud)
requirements.txt                Pinned, CPU-only torch build for reliable cloud builds
hr_pipeline/
  schema.py                     Detects & maps arbitrary column names to
                                 canonical fields (pros, cons, location, ...)
  clean.py                      Null normalization, dedup detection, date parsing
  sentiment.py                  Contextual RoBERTa sentiment engine + Mixed logic
  hr_dictionary.py               HR keyword/phrase matcher (topic tagging only)
  aggregate.py                  Category summary, positive drivers, risk signals,
                                 keyword analysis, coverage
  pipeline.py                   Orchestrates the above into one call
  benchmark.py                  Compares live output against a reference file
                                 (e.g. Bosch_Final_HR_Sentiment_Analysis_FINAL.xlsx)
assets/
  default_hr_dictionary.xlsx    Your 471-entry HR dictionary, used as the
                                 default if the evaluator doesn't upload one
tests/
  mock_model.py                 Offline stand-in used only to smoke-test the
                                 non-model parts of the pipeline during
                                 development. The deployed app always uses
                                 the real model — this file ships only so the
                                 test story is reproducible, and is never
                                 imported by app.py.
```

## Using the deployed app (evaluator's steps)

1. Open the URL.
2. Upload a CSV/Excel review export in the sidebar (optionally a custom HR
   dictionary too — the bundled one is used otherwise).
3. The backend validates columns, cleans the data, runs contextual
   sentiment analysis, tags HR themes, and renders the dashboard —
   automatically, no configuration.
4. Filter by whichever of Location, Employee Title, Employee Status,
   Employment Type, Tenure, Gender, HR Category, and Sentiment are present
   in the uploaded file — filters that don't apply (commonly Gender) are
   greyed out and explained rather than guessed at.
5. Explore all 7 tabs, including the category → keyword → review drill-down
   in HR Themes.

## Calibrating against your Colab analysis

You asked me to compare against `Bosch_Final_HR_Sentiment_Analysis_FINAL.xlsx`
*before* touching the 0.40 Mixed-meaningfulness threshold. Here's exactly
what I could and couldn't do, and why the threshold is unchanged.

**What I could check without the real model:** this sandbox has no network
access to huggingface.co, so I could not run the actual
`cardiffnlp/twitter-roberta-base-sentiment-latest` model here to get
apples-to-apples probabilities. What I *could* do is reverse-engineer
patterns from your `FINAL.xlsx`'s own numbers and check them against the
architecture this app uses:

- Back-calculating implied positive/negative probability from
  `sentiment_score`/`sentiment_magnitude` shows Mixed reviews are **not**
  simply "any review where both a Positive reason and a Negative reason
  string are present" — 545 reviews have both, but only 219 of those are
  labeled Mixed (284 are Positive, 21 Negative, 21 Neutral). That rules out
  a naive presence-based rule and supports this app's approach: a
  meaningfulness/confidence check per field, not bare text presence.
- Mixed reviews have consistently longer/more substantial `pros` **and**
  `cons` text (median 45/46 characters) than Positive or Negative reviews
  — consistent with "Mixed requires genuinely separate positive AND
  negative evidence," which is exactly what the 0.40 threshold is enforcing
  in this app's `hr_pipeline/sentiment.py`.
- I could not find a pattern in your reference file that specifically
  argues for a *different* number than 0.40 (e.g. 0.30 or 0.50) — every
  signal I could extract is directional, not precise enough to justify
  moving a specific cutoff without running the real model on the same
  text.

**What I did not find:** conclusive, review-by-review proof the app's
0.40 threshold over- or under-fires, since that requires the actual model's
output on actual text, which I can't produce here.

**Conclusion: the 0.40 threshold is unchanged**, per your instruction not
to move it without genuine evidence of a mismatch.

**Once deployed**, use the built-in benchmark tool to close this loop
yourself: upload your CSV to the app, then in the **Methodology / Data
Quality** tab, upload `Bosch_Final_HR_Sentiment_Analysis_FINAL.xlsx` to the
"Benchmark against a reference analysis" section. It reads the workbook's
`Employee Analysis` sheet automatically, aligns rows by `Review_ID` (or row
position as a fallback), and shows:
- overall agreement %,
- a full confusion matrix (reference label x this app's label),
- every mismatched review, so you can read the actual text and judge
  whether a given case is a real gap or a defensible difference in model
  behavior.

If that surfaces a genuine, repeatable mismatch pattern (not just a couple
of edge cases), the single constant to revisit is `MEANINGFUL_THRESHOLD` in
`hr_pipeline/sentiment.py` — change it there and redeploy.

## Sentiment methodology (unchanged from the original brief)

* **Sentiment Score** = Positive probability - Negative probability (-1..+1)
* **Sentiment Magnitude** = Positive probability + Negative probability (0..1)
* The star/overall rating is **never** read by the sentiment engine.
* `Pros` and `Cons` (and, with lower weight, the review title) are scored
  **separately** by the RoBERTa model. **Mixed** is only assigned when pros
  clears a real positive-confidence bar (`MEANINGFUL_THRESHOLD = 0.40`) *and*
  cons separately clears a real negative-confidence bar — not from a single
  ambivalent sentence, and not from bare presence of both a "Positive
  reason" and "Negative reason" string.
* Boilerplate like "no complaints", "nothing negative", "nothing much to
  say" is pattern-matched and excluded from counting as negative evidence,
  regardless of what words it contains — the same negation/context override
  from the original brief, applied unchanged.
* `sentiment_reason` is built from the review's own pros/cons/title text —
  never invented.

## HR dictionary matching

Word-boundary-aware matching (keyword or phrase + its listed variant)
against Pros/Cons/Title/Advice-to-management text. It only tags **which**
HR category/keyword a review touches — it never overrides the contextual
sentiment label above. Reviews matching nothing get
`"No specific HR theme identified"` rather than a guessed category.

## Data validation & no-hallucination guarantees

* If no usable review-text column exists at all, the app stops and tells
  the evaluator exactly what's missing rather than guessing.
* If an optional field (gender, tenure, location, ...) is absent, the
  matching filter/analysis is disabled and clearly labeled — gender is
  never inferred.
* Every statistic on every tab is computed live from the uploaded file (and
  recomputed against whatever filters are active) — nothing is hard-coded
  to the Bosch reference numbers (977 reviews / 471 dictionary entries /
  ~85-86% coverage). Those only reappear if the Bosch file is literally
  re-uploaded. Verified during development against a synthetic
  differently-shaped "Dell-style" export (different column names/order, no
  gender column) — same pipeline, zero code changes, correct results.
* A keyword flagged in "HR Risk / Mixed Signals" is explicitly labeled
  **Low-volume signal** rather than a stated organizational problem when
  its sample size is small.

## Local development (optional, not required for the evaluator)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
