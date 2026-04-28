"""
evaluate_models.py
═══════════════════════════════════════════════════════════════════════
Evaluation harness for HotpotQA and MuSiQue benchmarks.

Wired to the COM726 LangGraph project's `llms` module so every model
is called exactly as the application calls it (llms.build_llm), with
NO dependency on Streamlit, MySQL, LangGraph, or ChromaDB.

Computes for every model:
  • EM             – Exact Match (%)
  • F1             – Token F1   (%)
  • Retrieval P@4  – Precision@4 (fraction of runs where ALL gold
                     supporting titles appear in the top-4 retrieved
                     passages chosen by the BM25 retriever below)
  • Latency        – Average inference latency (ms)

Results are saved to:
  testing_data/hotpotqa_results.json
  testing_data/musique_results.json

HOW TO RUN
──────────
  # Can be run from ANY directory — the script finds the project root
  # automatically by searching for the `llms/` package folder:
  python evaluate_models.py
  python tests/evaluate_models.py

  # Limit questions for a quick smoke-test:
  N_QUESTIONS = 20   ← change the constant below
═══════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import json
import math
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable, TypedDict


# ── Auto-discover project root: needs BOTH llms/ AND config/ ─────────
def _find_project_root() -> Path:
    """
    Walk upward from this file until we find a directory containing
    BOTH an `llms/` package and a `config/` package/folder.
    This ensures we land on `langgraph_chatbot/` and not a parent
    directory that happens to have `llms/` via a nested project.
    Falls back to searching for `llms/` alone if config is absent.
    """
    start = Path(__file__).resolve().parent

    # Pass 1: both llms/ AND config/ present — most specific match
    candidate = start
    for _ in range(8):
        has_llms = (candidate / "llms" / "__init__.py").exists()
        has_config = (
            (candidate / "config" / "__init__.py").exists()
            or (candidate / "config" / "settings.py").exists()
        )
        if has_llms and has_config:
            return candidate
        candidate = candidate.parent

    # Pass 2: llms/ alone — weaker fallback
    candidate = start
    for _ in range(8):
        if (candidate / "llms" / "__init__.py").exists():
            return candidate
        candidate = candidate.parent

    # Give up — return the script's own dir
    return start


_ROOT = _find_project_root()
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

print(f"[evaluate_models] Project root resolved to: {_ROOT}")

# ── Dataset loader (pip install datasets) ────────────────────────────
try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "The `datasets` package is required.\n"
        "Run:  pip install datasets\n"
        f"Original error: {exc}"
    ) from exc

# ── Project LLM interface ─────────────────────────────────────────────
try:
    import llms
except ImportError as exc:
    raise SystemExit(
        f"Could not import the project's `llms` package.\n"
        f"Searched for llms/__init__.py starting from: {Path(__file__).resolve().parent}\n"
        f"Resolved project root to: {_ROOT}\n"
        f"Current sys.path: {sys.path[:5]}\n"
        f"Original error: {exc}"
    ) from exc


# ─────────────────────────────────────────────────────────────────────
#  CONSTANTS  — adjust as needed
# ─────────────────────────────────────────────────────────────────────

DATA_DIR = "testing_data"
N_QUESTIONS = 500       # questions per benchmark per model
RETRIEVAL_K = 4         # P@K — top-k passages passed to the model

# Prompt template used for every model call
_QA_PROMPT = (
    "You are a precise question-answering assistant.\n"
    "Use ONLY the context passages below to answer. "
    "Reply with a short, direct answer — a few words or one sentence.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ─────────────────────────────────────────────────────────────────────
#  MODEL KEY → display name map
#  Keys must exactly match llms.list_model_keys() values in your project
# ─────────────────────────────────────────────────────────────────────

TOKEN_LIMITS: dict[str, int] = {
    "llama-8b-instant":   4000,
    # "gemini-2.5-flash":   8000,
    "llama3.2-3b":        3000,
    "qwen3.5-0.8b":       3000,
    "phi3-3.8b":          3000,
    "granite3-dense-2b":  3000,
    "gemma3-270m":        2000,
    "qwen2_5_coder_7b":   4000,
    "deepseek_r1":        8000,
    "falcon3":            3000,
    "mistral-7b":         4000,
}

# Maps the table row label → llms model key
# Adjust if your llms.list_model_keys() returns different strings
MODEL_KEY_MAP: dict[str, str] = {
    "deepseek_r1":          "deepseek_r1",
    "falcon3":              "falcon3",
    # "gemini_2_5_flash":     "gemini-2.5-flash",
    "gemma3_270m":          "gemma3-270m",
    "granite3_dense_2b":    "granite3-dense-2b",
    "llama_3_1_8b_instant": "llama-8b-instant",
    "mistral_7b":           "mistral-7b",
    "phi3_3_8b":            "phi3-3.8b",
    "qwen2_5_coder_7b":     "qwen2_5_coder_7b",
    "qwen3_5_0_8b":         "qwen3.5-0.8b",
}


# ─────────────────────────────────────────────────────────────────────
#  TYPE DEFINITIONS
# ─────────────────────────────────────────────────────────────────────

class ModelOutput(TypedDict):
    answer:    str        # model's answer string
    retrieved: list[str]  # titles of the top-k passages used


ModelFn = Callable[[str, list[str]], ModelOutput]


# ─────────────────────────────────────────────────────────────────────
#  LIGHTWEIGHT BM25 RETRIEVER
#  — no external library required; gives us titles for P@4 scoring
# ─────────────────────────────────────────────────────────────────────

def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return text.split()


def _bm25_retrieve(
    query: str,
    passages: list[str],   # each passage is "Title: body text"
    k: int = RETRIEVAL_K,
) -> tuple[list[str], list[str]]:
    """
    BM25 retrieval over `passages`.

    Returns
    -------
    top_passages : list[str]   — top-k full passage strings (for the LLM prompt)
    top_titles   : list[str]   — titles extracted from those passages
    """
    # BM25 hyper-parameters
    K1, B = 1.5, 0.75

    # Build corpus token lists
    corpus = [_tokenise(p) for p in passages]
    avgdl = sum(len(d) for d in corpus) / max(len(corpus), 1)

    # Build IDF table
    df: Counter = Counter()
    for doc in corpus:
        for term in set(doc):
            df[term] += 1
    N = len(corpus)

    def idf(term: str) -> float:
        n = df.get(term, 0)
        return math.log((N - n + 0.5) / (n + 0.5) + 1.0)

    # Score each passage
    query_terms = _tokenise(query)
    scores: list[float] = []
    for doc in corpus:
        tf_map = Counter(doc)
        dl = len(doc)
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            denom = tf + K1 * (1 - B + B * dl / max(avgdl, 1))
            score += idf(term) * (tf * (K1 + 1)) / max(denom, 1e-9)
        scores.append(score)

    # Pick top-k indices
    top_indices = sorted(range(len(scores)),
                         key=lambda i: scores[i], reverse=True)[:k]

    top_passages = [passages[i] for i in top_indices]

    # Extract title: the part before the first ": "
    def _extract_title(passage: str) -> str:
        if ": " in passage:
            return passage.split(": ", 1)[0].strip()
        return passage[:60]

    top_titles = [_extract_title(p) for p in top_passages]
    return top_passages, top_titles


# ─────────────────────────────────────────────────────────────────────
#  MODEL CALLABLE FACTORY
#  Wraps llms.build_llm(model_key) into the ModelFn signature
# ─────────────────────────────────────────────────────────────────────

def _make_model_fn(llm_key: str) -> ModelFn:
    """
    Returns a callable that:
      1. BM25-retrieves the top-4 passages from context
      2. Builds a concise QA prompt
      3. Invokes the model via llms.build_llm(llm_key)
      4. Returns ModelOutput { answer, retrieved }
    """
    def fn(question: str, context: list[str]) -> ModelOutput:
        # ── Retrieve top-k passages ───────────────────────────────
        top_passages, top_titles = _bm25_retrieve(
            question, context, k=RETRIEVAL_K)

        # ── Build prompt ─────────────────────────────────────────
        context_block = "\n\n".join(
            f"[{i+1}] {p}" for i, p in enumerate(top_passages)
        )
        prompt = _QA_PROMPT.format(
            context=context_block,
            question=question,
        )

        # ── Call the model ────────────────────────────────────────
        lm = llms.build_llm(llm_key)
        response = lm.invoke(prompt)

        # langchain LLMs return an object with .content; plain string otherwise
        answer = response.content if hasattr(
            response, "content") else str(response)
        answer = answer.strip()

        return ModelOutput(answer=answer, retrieved=top_titles)

    return fn


# ─────────────────────────────────────────────────────────────────────
#  ADAPTIVE MODEL CALLABLE
#  Uses the orchestrator (same logic as _call_llm_with_orchestrator
#  in graph.py) but without any Streamlit / MySQL / LangGraph deps
# ─────────────────────────────────────────────────────────────────────

def _make_adaptive_fn() -> ModelFn:
    """
    Adaptive (Ours): uses the orchestrator to auto-select the best
    model per question, mirroring direct_node in graph.py.
    Falls back to the default model if the orchestrator is unavailable.
    """
    try:
        from llms.custom.orchestrator import (
            classify_task,
            select_primary_model,
            build_specialist_prompt,
        )
        _has_orchestrator = True
    except ImportError:
        _has_orchestrator = False
        print("[Adaptive] orchestrator not importable — will use default model.")

    # config.settings lives in the project root; guard against import
    # errors so the script still runs even if config can't be loaded
    try:
        from config.settings import DEFAULT_MODEL_KEY  # type: ignore
    except ModuleNotFoundError:
        DEFAULT_MODEL_KEY = "llama-8b-instant"  # safe project default
        print(
            "[Adaptive] config.settings not importable — "
            f"using built-in default model: {DEFAULT_MODEL_KEY}"
        )

    def fn(question: str, context: list[str]) -> ModelOutput:
        top_passages, top_titles = _bm25_retrieve(
            question, context, k=RETRIEVAL_K)
        context_block = "\n\n".join(
            f"[{i+1}] {p}" for i, p in enumerate(top_passages)
        )

        if _has_orchestrator:
            categories, complexity = classify_task(question)
            model_key = select_primary_model(categories, complexity)
            prompt = build_specialist_prompt(
                question, categories, complexity, model_key)
            # Prepend retrieved context
            prompt = (
                f"CONTEXT FROM DOCUMENTS:\n{context_block}\n\n"
                f"Answer the following question using ONLY the context above.\n\n"
                + prompt
            )
        else:
            model_key = DEFAULT_MODEL_KEY
            prompt = _QA_PROMPT.format(
                context=context_block, question=question)

        lm = llms.build_llm(model_key)
        response = lm.invoke(prompt)
        answer = response.content if hasattr(
            response, "content") else str(response)
        return ModelOutput(answer=answer.strip(), retrieved=top_titles)

    return fn


# ─────────────────────────────────────────────────────────────────────
#  DISCOVER AVAILABLE MODELS AT RUNTIME
#  Falls back to MODEL_KEY_MAP if llms.list_model_keys() fails
# ─────────────────────────────────────────────────────────────────────

def _build_model_fn_map() -> dict[str, ModelFn]:
    """
    Build MODEL_FN_MAP dynamically from the project's registered models.
    Only includes models that are actually available via llms.list_model_keys().
    """
    try:
        available_keys = set(llms.list_model_keys())
    except Exception as exc:
        print(f"[WARNING] llms.list_model_keys() failed: {exc}")
        print("          Falling back to MODEL_KEY_MAP constants.")
        available_keys = set(MODEL_KEY_MAP.values())

    fn_map: dict[str, ModelFn] = {}
    skipped: list[str] = []

    for label, llm_key in MODEL_KEY_MAP.items():
        if llm_key in available_keys:
            fn_map[label] = _make_model_fn(llm_key)
        else:
            skipped.append(f"{label} ({llm_key})")

    if skipped:
        print(
            f"[INFO] The following models are NOT registered in llms and will be "
            f"skipped:\n       " + "\n       ".join(skipped)
        )

    # Always include Adaptive regardless — it uses whichever model the
    # orchestrator selects dynamically
    fn_map["Adaptive (Ours)"] = _make_adaptive_fn()

    return fn_map


# ─────────────────────────────────────────────────────────────────────
#  METRIC HELPERS
# ─────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalise(prediction) == _normalise(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = _normalise(prediction).split()
    gold_tokens = _normalise(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def retrieval_precision_at_k(
    retrieved_titles: list[str],
    gold_titles:      list[str],
    k:                int = RETRIEVAL_K,
) -> float:
    """
    Hard P@K: 1.0 if ALL gold supporting titles appear in top-k, else 0.0.
    For soft (fraction), replace the return with: return hits / len(gold_titles)
    """
    top_k = {t.lower() for t in retrieved_titles[:k]}
    hits = sum(1 for g in gold_titles if g.lower() in top_k)
    return float(hits == len(gold_titles))


# ─────────────────────────────────────────────────────────────────────
#  DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────

def load_hotpotqa(n: int = N_QUESTIONS) -> list[dict]:
    """
    HotpotQA distractor split — validation set.

    Returns list of dicts:
        question        : str
        answer          : str
        context_texts   : list[str]   "Title: passage"
        gold_titles     : list[str]   supporting fact titles
    """
    print("  Downloading / loading HotpotQA …", flush=True)
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    samples: list[dict] = []
    for row in ds.select(range(min(n, len(ds)))):
        titles = row["context"]["title"]
        sents = row["context"]["sentences"]
        passages = [" ".join(s_list) for s_list in sents]
        context_texts = [f"{t}: {p}" for t, p in zip(titles, passages)]
        gold_titles = list({sf[0] for sf in row["supporting_facts"]["title"]})
        samples.append({
            "question":      row["question"],
            "answer":        row["answer"],
            "context_texts": context_texts,
            "gold_titles":   gold_titles,
        })
    return samples


def load_musique(n: int = N_QUESTIONS) -> list[dict]:
    """
    MuSiQue answerable validation split.

    Returns the same dict structure as load_hotpotqa.
    """
    print("  Downloading / loading MuSiQue …", flush=True)

    try:
        # ✅ FIX 1: Correct dataset name
        ds = load_dataset("musique", split="validation")

    except Exception:
        try:
            # ✅ FIX 2: fallback config (some versions require this)
            ds = load_dataset("musique", "all", split="validation")
        except Exception as exc:
            raise RuntimeError(
                "Could not load MuSiQue dataset.\n"
                "Tried:\n"
                "  - load_dataset('musique')\n"
                "  - load_dataset('musique', 'all')\n\n"
                "Fixes to try:\n"
                "  pip install -U datasets\n"
                f"\nOriginal error: {exc}"
            ) from exc

    # ✅ FIX 3: safer filtering
    answerable = [r for r in ds if r.get("answerable", True)]

    samples: list[dict] = []

    for row in answerable[:n]:
        paragraphs = row.get("paragraphs", [])

        # ✅ FIX 4: robust paragraph text extraction
        def _body(p: dict) -> str:
            return (
                p.get("paragraph_text")
                or p.get("body")
                or p.get("text")
                or ""
            )

        context_texts = [
            f"{p.get('title', 'Unknown')}: {_body(p)}"
            for p in paragraphs
        ]

        gold_titles = [
            p.get("title", "")
            for p in paragraphs
            if p.get("is_supporting")
        ]

        samples.append({
            "question":      row.get("question", ""),
            "answer":        row.get("answer", ""),
            "context_texts": context_texts,
            "gold_titles":   gold_titles,
        })

    return samples


# ─────────────────────────────────────────────────────────────────────
#  CORE EVALUATION LOOP
# ─────────────────────────────────────────────────────────────────────

def evaluate_model(
    model_key: str,
    model_fn:  ModelFn,
    samples:   list[dict],
) -> dict:
    """
    Runs model_fn over every sample and returns aggregated metrics.
    """
    em_scores: list[float] = []
    f1_scores: list[float] = []
    p4_scores: list[float] = []
    latencies: list[float] = []

    for i, sample in enumerate(samples, 1):
        print(f"   [{i:>4}/{len(samples)}]", end="\r", flush=True)

        t_start = time.perf_counter()
        try:
            output: ModelOutput = model_fn(
                sample["question"],
                sample["context_texts"],
            )
        except NotImplementedError:
            print(
                f"\n   ⚠  [{model_key}] not implemented — skipping all samples.")
            break
        except Exception as exc:
            print(f"\n   ⚠  Sample {i} failed: {exc}")
            continue
        latency_ms = (time.perf_counter() - t_start) * 1000

        pred = output["answer"]
        retrieved = output["retrieved"]
        gold_ans = sample["answer"]
        gold_ttl = sample["gold_titles"]

        em_scores.append(exact_match(pred, gold_ans))
        f1_scores.append(token_f1(pred, gold_ans))
        p4_scores.append(retrieval_precision_at_k(retrieved, gold_ttl))
        latencies.append(latency_ms)

    def _mean_pct(lst: list[float]) -> float:
        return round(sum(lst) / len(lst) * 100, 2) if lst else 0.0

    def _mean(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "em":             _mean_pct(em_scores),
        "f1":             _mean_pct(f1_scores),
        "retrieval_p4":   _mean(p4_scores),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "n_evaluated":    len(em_scores),
    }


def run_benchmark(
    benchmark_name: str,
    samples:        list[dict],
    output_file:    str,
    model_fn_map:   dict[str, ModelFn],
) -> None:
    """Iterates over all models, evaluates, and saves results to JSON."""
    print(f"\n{'═' * 62}")
    print(f"  Benchmark : {benchmark_name}  ({len(samples)} questions)")
    print(f"{'═' * 62}")

    results: dict[str, dict] = {}

    for model_key, model_fn in model_fn_map.items():
        print(f"\n  ▶ {model_key}")
        metrics = evaluate_model(model_key, model_fn, samples)
        results[model_key] = metrics
        print(
            f"\n     EM={metrics['em']:.2f}%  "
            f"F1={metrics['f1']:.2f}%  "
            f"P@{RETRIEVAL_K}={metrics['retrieval_p4']:.4f}  "
            f"Lat={metrics['avg_latency_ms']:.1f}ms  "
            f"(n={metrics['n_evaluated']})"
        )

    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, output_file)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "benchmark":   benchmark_name,
                "n_questions": N_QUESTIONS,
                "results":     results,
            },
            fh,
            indent=2,
        )
    print(f"\n  [✓] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building model function map …")
    MODEL_FN_MAP = _build_model_fn_map()
    print(f"  {len(MODEL_FN_MAP)} models registered: {', '.join(MODEL_FN_MAP)}\n")

    print("Loading datasets …")
    hotpotqa_samples = load_hotpotqa(N_QUESTIONS)
    musique_samples = load_musique(N_QUESTIONS)
    print(f"  HotpotQA : {len(hotpotqa_samples)} samples")
    print(f"  MuSiQue  : {len(musique_samples)}  samples")

    run_benchmark(
        benchmark_name="HotpotQA",
        samples=hotpotqa_samples,
        output_file="hotpotqa_results.json",
        model_fn_map=MODEL_FN_MAP,
    )

    run_benchmark(
        benchmark_name="MuSiQue",
        samples=musique_samples,
        output_file="musique_results.json",
        model_fn_map=MODEL_FN_MAP,
    )

    print("\n[Done] All results saved to testing_data/\n")
