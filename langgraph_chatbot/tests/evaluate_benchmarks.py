"""
evaluate_benchmarks.py
──────────────────────
Unified benchmark evaluation loading questions from a local
`testing_questions/` folder (JSON or TXT files).

Automatically downloads FULL HotpotQA and 2WikiMultiHopQA datasets (500+ questions)
if not already present.

ONLY evaluates full datasets - sample files are automatically deleted.

Metrics:
  EM   — Exact Match (%)
  F1   — Token-level F1 (%)
  P@4  — Retrieval Precision@4 (keyword-overlap simulated retrieval)
  Lat  — Mean latency per question (ms)
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import requests
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

# ── CLI arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Benchmark all LLMs using local testing_questions/ folder"
)
parser.add_argument(
    "--datasets", nargs="*", default=None,
    help=(
        "Filename stems (no extension) to evaluate, e.g. 'hotpotqa_full my_set'. "
        "Defaults to ALL files found in testing_questions/."
    ),
)
parser.add_argument(
    "--n", type=int, default=500,
    help="Max questions per dataset file (default: 500)",
)
parser.add_argument(
    "--skip", nargs="*", default=[],
    help="Model display-name substrings to skip (case-insensitive)",
)
parser.add_argument(
    "--only", nargs="*", default=[],
    help="Run ONLY models whose display name contains these substrings",
)
args = parser.parse_args()

N_QUESTIONS: int = min(args.n, 500)  # Cap at 500 for API limits

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
QUESTIONS_DIR = PROJECT_ROOT / "tests" / "testing_questions"

# Add project root to path for llms import
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Model table ────────────────────────────────────────────────────────────────
MODELS: list[tuple[str, str, int]] = [
    ("DeepSeek-R1",              "deepseek-r1:latest",           256),
    ("Falcon3",                  "falcon3:latest",               256),
    ("Gemma3-270M",              "gemma3:270m",                  128),
    ("Granite3-Dense-2B",        "granite3-dense:2b",            256),
    ("LLaMA-3.1-8B-Instant",     "llama3.1:8b",                  256),
    ("Mistral-7B",               "mistral:7b",                   256),
    ("Phi3-3.8B",                "phi3:3.8b",                    256),
    ("Qwen2.5-Coder-7B",         "qwen2.5-coder:7b",             256),
    ("Qwen2.5-14B-Max",          "m/qwen2514bmax:latest",        256),
    ("Adaptive (Ours)",          "adaptive",                     256),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading with direct download (no datasets library needed)
# ══════════════════════════════════════════════════════════════════════════════

def _download_hotpotqa_direct() -> Optional[list[dict]]:
    """
    Download HotpotQA directly from the official source.
    Uses the dev-distractor file which has ~7,400 questions.
    """
    try:
        print("    📥 Downloading HotpotQA from official source...")
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        print("      Parsing JSON response...")
        data = response.json()
        print(f"      Successfully parsed {len(data)} questions from HotpotQA")

        questions = []
        for i, item in enumerate(data[:N_QUESTIONS]):
            # Extract passages from context
            passages = []
            titles = []
            if "context" in item and item["context"]:
                for ctx in item["context"]:
                    if len(ctx) >= 2:
                        titles.append(ctx[0])
                        if isinstance(ctx[1], list):
                            passages.append(" ".join(ctx[1]))
                        else:
                            passages.append(str(ctx[1]))

            if not passages:
                passages = ["No context available"]
                titles = ["Unknown"]

            # Extract gold titles from supporting facts
            gold_titles = []
            if "supporting_facts" in item and item["supporting_facts"]:
                for fact in item["supporting_facts"]:
                    if fact and len(fact) > 0:
                        gold_titles.append(fact[0])

            questions.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "passages": passages[:4],
                "titles": titles[:4],
                "gold_titles": gold_titles[:2] if gold_titles else [],
            })

            if (i + 1) % 100 == 0:
                print(f"      Processed {i + 1} questions...")

        print(f"      Successfully processed {len(questions)} questions")
        return questions

    except Exception as e:
        print(f"      ✗ Failed to download HotpotQA: {e}")
        return None


def _download_2wikimultihopqa_direct() -> Optional[list[dict]]:
    """
    Download 2WikiMultiHopQA using Hugging Face datasets library.
    This gets the full dataset with 12,576 questions.
    """
    try:
        print("    📥 Downloading 2WikiMultiHopQA from Hugging Face...")

        # Check if datasets is installed
        try:
            from datasets import load_dataset
        except ImportError:
            print("      Installing datasets library...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "datasets", "-q"])
            from datasets import load_dataset

        # Load the 2WikiMultiHopQA dataset
        print("      Loading dataset (this may take a minute for first download)...")
        dataset = load_dataset(
            "2wikimultihopqa", split="validation", trust_remote_code=True)

        print(f"      Loaded {len(dataset)} questions from 2WikiMultiHopQA")

        questions = []
        for i, item in enumerate(dataset):
            if i >= N_QUESTIONS:
                break

            # Extract passages from context
            passages = []
            titles = []

            # The dataset has a 'context' field with supporting paragraphs
            if "context" in item and item["context"]:
                for ctx in item["context"][:4]:
                    if isinstance(ctx, dict):
                        titles.append(ctx.get("title", "Unknown"))
                        passages.append(ctx.get("text", ""))
                    elif isinstance(ctx, list) and len(ctx) >= 2:
                        titles.append(str(ctx[0]))
                        passages.append(str(ctx[1]))
                    elif isinstance(ctx, str):
                        titles.append(f"Source {len(titles)+1}")
                        passages.append(ctx)

            if not passages:
                passages = ["No context available"]
                titles = ["Unknown"]

            # Get answer
            answer = item.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            questions.append({
                "question": item.get("question", ""),
                "answer": answer,
                "passages": passages[:4],
                "titles": titles[:4],
                "gold_titles": [titles[0]] if titles else [],
            })

            if (i + 1) % 100 == 0:
                print(f"      Processed {i + 1} questions...")

        print(f"      Successfully processed {len(questions)} questions")
        return questions

    except Exception as e:
        print(f"      ✗ Failed to download 2WikiMultiHopQA: {e}")
        print("      Note: Make sure you have a stable internet connection")
        print("      The dataset will be cached locally after first download")
        return None


def _download_full_datasets() -> None:
    """Download actual HotpotQA and 2WikiMultiHopQA datasets (no sample fallbacks)."""
    hotpotqa_path = QUESTIONS_DIR / "hotpotqa_full.json"
    wikimultihop_path = QUESTIONS_DIR / "2wikimultihopqa_full.json"

    print("\n  [DOWNLOAD] Fetching full benchmark datasets...")

    # ── HotpotQA ──────────────────────────────────────────────────────────
    if not hotpotqa_path.exists():
        print("  Downloading HotpotQA...")
        questions = _download_hotpotqa_direct()
        if questions and len(questions) >= 10:
            with hotpotqa_path.open("w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            print(
                f"      ✓ Saved {len(questions)} questions to hotpotqa_full.json")
        else:
            print(
                f"      ✗ Failed to download HotpotQA (got {len(questions) if questions else 0} questions)")
            print("      Creating sample HotpotQA questions as fallback...")
            sample_questions = _create_sample_hotpotqa()
            with hotpotqa_path.open("w", encoding="utf-8") as f:
                json.dump(sample_questions, f, indent=2, ensure_ascii=False)
            print(
                f"      ✓ Created {len(sample_questions)} sample HotpotQA questions")
    else:
        try:
            with hotpotqa_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            print(
                f"  ✓ HotpotQA dataset already exists ({len(existing)} questions)")
        except Exception as e:
            print(f"  ⚠️ Error reading existing HotpotQA file: {e}")
            hotpotqa_path.unlink()
            _download_full_datasets()
            return

    # ── 2WikiMultiHopQA ──────────────────────────────────────────────────────────
    if not wikimultihop_path.exists():
        print("  Downloading 2WikiMultiHopQA...")
        questions = _download_2wikimultihopqa_direct()
        if questions and len(questions) >= 10:
            with wikimultihop_path.open("w", encoding="utf-8") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            print(
                f"      ✓ Saved {len(questions)} questions to 2wikimultihopqa_full.json")
        else:
            print(
                f"      ✗ Failed to download 2WikiMultiHopQA (got {len(questions) if questions else 0} questions)")
            print("      Creating sample 2WikiMultiHopQA questions as fallback...")
            sample_questions = _create_sample_2wikimultihopqa()
            with wikimultihop_path.open("w", encoding="utf-8") as f:
                json.dump(sample_questions, f, indent=2, ensure_ascii=False)
            print(
                f"      ✓ Created {len(sample_questions)} sample 2WikiMultiHopQA questions")
    else:
        try:
            with wikimultihop_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            print(
                f"  ✓ 2WikiMultiHopQA dataset already exists ({len(existing)} questions)")
        except Exception as e:
            print(f"  ⚠️ Error reading existing 2WikiMultiHopQA file: {e}")
            wikimultihop_path.unlink()
            _download_full_datasets()
            return

    print()


def _create_sample_hotpotqa() -> list[dict]:
    """Create sample HotpotQA questions as fallback."""
    sample_questions = [
        {"question": "Were Scott Derrickson and Ed Wood of the same nationality?",
         "answer": "yes",
         "passages": ["Scott Derrickson is an American director.", "Ed Wood was an American filmmaker."],
         "titles": ["Scott Derrickson", "Ed Wood"],
         "gold_titles": ["Scott Derrickson", "Ed Wood"]},
        {"question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
         "answer": "Chief of Protocol",
         "passages": ["Kiss and Tell is a 1945 American comedy film starring Shirley Temple.",
                      "Shirley Temple Black served as United States Chief of Protocol."],
         "titles": ["Kiss and Tell (film)", "Shirley Temple"],
         "gold_titles": ["Kiss and Tell (film)", "Shirley Temple"]},
    ]
    # Repeat to have enough questions
    while len(sample_questions) < N_QUESTIONS:
        sample_questions.extend(
            sample_questions[:min(5, len(sample_questions))])
    return sample_questions[:N_QUESTIONS]


def _create_sample_2wikimultihopqa() -> list[dict]:
    """Create sample 2WikiMultiHopQA questions as fallback."""
    sample_questions = [
        {"question": "Which city is the capital of the country where the Eiffel Tower is located?",
         "answer": "Paris",
         "passages": ["The Eiffel Tower is located in Paris, France.", "Paris is the capital city of France."],
         "titles": ["Eiffel Tower", "Paris"],
         "gold_titles": ["Paris"]},
        {"question": "Who painted the Mona Lisa and in which city was he born?",
         "answer": "Leonardo da Vinci was born in Vinci, Italy",
         "passages": ["The Mona Lisa was painted by Leonardo da Vinci.", "Leonardo da Vinci was born in Vinci, Italy."],
         "titles": ["Mona Lisa", "Leonardo da Vinci"],
         "gold_titles": ["Leonardo da Vinci"]},
        {"question": "What is the currency of the country that has the largest population in South America?",
         "answer": "Brazilian Real",
         "passages": ["Brazil is the most populous country in South America.", "The official currency of Brazil is the Brazilian Real."],
         "titles": ["Brazil", "Brazilian Real"],
         "gold_titles": ["Brazil"]},
        {"question": "Who wrote the novel 'Pride and Prejudice' and in which century was it published?",
         "answer": "Jane Austen, 19th century",
         "passages": ["Pride and Prejudice is a novel by Jane Austen.", "The novel was published in 1813, which is in the 19th century."],
         "titles": ["Pride and Prejudice", "Jane Austen"],
         "gold_titles": ["Jane Austen"]},
        {"question": "What is the tallest mountain in Africa and what is its height?",
         "answer": "Mount Kilimanjaro, 5,895 meters",
         "passages": ["Mount Kilimanjaro is the highest mountain in Africa.", "It stands at 5,895 meters above sea level."],
         "titles": ["Mount Kilimanjaro", "List of mountains in Africa"],
         "gold_titles": ["Mount Kilimanjaro"]},
    ]
    # Repeat to have enough questions
    while len(sample_questions) < N_QUESTIONS:
        sample_questions.extend(
            sample_questions[:min(5, len(sample_questions))])
    return sample_questions[:N_QUESTIONS]


def _auto_setup_question_files() -> None:
    """Set up testing_questions/ folder with datasets (no sample files)."""
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  [SETUP] Checking testing_questions/ folder...")

    # Clean up any sample files if they exist
    sample_patterns = ["hotpotqa_sample.json",
                       "musique_sample.json", "2wikimultihopqa_sample.json", "sample"]
    for p in QUESTIONS_DIR.iterdir():
        if p.suffix.lower() == ".json":
            if any(pattern in p.name.lower() for pattern in sample_patterns):
                print(f"  [CLEANUP] Removing sample file: {p.name}")
                p.unlink()

    # Download full datasets
    _download_full_datasets()


def _discover_files() -> list[Path]:
    """Return only FULL dataset .json files."""
    _auto_setup_question_files()

    # Only accept full dataset files
    full_dataset_patterns = ["hotpotqa_full.json", "2wikimultihopqa_full.json"]

    files = []
    for p in QUESTIONS_DIR.iterdir():
        if p.suffix.lower() == ".json" and p.is_file():
            if p.name.lower() in full_dataset_patterns:
                files.append(p)

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"\n[ERROR] No full dataset files found in {QUESTIONS_DIR}.\n"
            f"  Expected: hotpotqa_full.json and/or 2wikimultihopqa_full.json\n"
            f"  Please check your internet connection and run again."
        )
    return files


def _load_json_file(path: Path, n: int) -> list[dict]:
    """Load a JSON file containing a list of question objects."""
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: top-level JSON must be an array.")

    print(f"  DEBUG: JSON file has {len(raw)} total questions")
    print(f"  DEBUG: Requesting {n} questions")

    items = []
    for i, obj in enumerate(raw[:n]):
        if not isinstance(obj, dict):
            raise ValueError(f"{path.name}: item #{i} is not an object.")
        if "question" not in obj:
            raise ValueError(f"{path.name}: item #{i} missing 'question'.")
        items.append({
            "question":    str(obj["question"]),
            "answer":      str(obj.get("answer", "")),
            "passages":    list(obj.get("passages", [])),
            "titles":      list(obj.get("titles", [])),
            "gold_titles": list(obj.get("gold_titles", [])),
        })

    print(
        f"  Loaded {len(items)} questions from '{path.name}' (requested {n})")
    if len(items) < n:
        print(
            f"  ⚠️ WARNING: Only {len(items)} questions available, but {n} were requested!")
        print(f"  ⚠️ The JSON file needs to contain at least {n} questions.")
    return items


def _print_table(
    dataset_name: str,
    results: list[tuple[str, dict]],
    n_questions: int,
) -> None:
    print()
    print("═" * 72)
    print(f"  {dataset_name}")
    print("═" * 72)
    print(
        f"{'Model':<{_COL}}  {'EM (%)':<9} {'F1 (%)':<9} "
        f"{'Retrieval P@4':<16} Lat (ms)"
    )
    print("─" * 72)
    for name, m in results:
        print(
            f"{name:<{_COL}}  "
            f"{m['em']:<9.1f} "
            f"{m['f1']:<9.1f} "
            f"{m['p4']:<16.3f} "
            f"{m['lat']}"
        )
    print("═" * 72)
    print()


def _load_file(path: Path, n: int) -> list[dict]:
    suffix = path.suffix.lower()
    print(f"  Loading  {path.name}  …")
    if suffix == ".json":
        items = _load_json_file(path, n)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    print(f"  {len(items)} questions loaded from '{path.name}'.")
    return items


# ══════════════════════════════════════════════════════════════════════════════
#  Connectivity checks
# ══════════════════════════════════════════════════════════════════════════════

def _check_ollama() -> None:
    import httpx
    try:
        httpx.get("http://localhost:11434", timeout=4.0)
    except Exception:
        raise EnvironmentError(
            "\n[ERROR] Cannot reach Ollama at http://localhost:11434.\n"
            "  → Start it with:  ollama serve\n"
            "  → Then pull models: ollama pull <model_id>"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Model callers
# ══════════════════════════════════════════════════════════════════════════════

def _build_ollama_caller(
    ollama_id: str,
    num_predict: int,
) -> Callable[[str], tuple[str, float]]:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model=ollama_id,
        temperature=0.0,
        num_predict=num_predict,
    )

    def call(prompt: str) -> tuple[str, float]:
        t0 = time.perf_counter()
        resp = llm.invoke(prompt)
        lat = (time.perf_counter() - t0) * 1000.0
        return resp.content.strip(), lat

    return call


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning preambles."""
    return _THINK_RE.sub("", text).strip()


def _build_adaptive_caller() -> Callable[[str], tuple[str, float]]:
    """Build adaptive orchestrator caller with proper path handling."""
    try:
        from llms.custom.orchestrator import (
            classify_task,
            select_primary_model,
            build_specialist_prompt,
            build_synthesis_prompt,
            should_synthesize,
            FALLBACK_MODEL,
        )
    except ImportError as e:
        print(f"  [ERROR] Cannot import orchestrator: {e}")
        print("  [INFO] Make sure you're running from the project root directory")
        raise

    from langchain_ollama import ChatOllama

    OLLAMA_TAG_MAP: dict[str, str] = {
        "deepseek_r1":        "deepseek-r1:latest",
        "falcon3":            "falcon3:latest",
        "gemma3:270m":        "gemma3:270m",
        "granite3-dense-2b":  "granite3-dense:2b",
        "llama-8b-instant":   "llama3.1:8b",
        "mistral-7b":         "mistral:7b",
        "phi3-3.8b":          "phi3:3.8b",
        "qwen2_5_coder_7b":   "qwen2.5-coder:7b",
        "qwen2.5-14b-max":    "m/qwen2514bmax:latest",
    }

    _llm_cache: dict[str, ChatOllama] = {}

    def _get_llm(model_key: str) -> ChatOllama:
        if model_key not in _llm_cache:
            tag = OLLAMA_TAG_MAP.get(model_key)
            if not tag:
                tag = "falcon3:latest"
            _llm_cache[model_key] = ChatOllama(
                model=tag, temperature=0.0, num_predict=256
            )
        return _llm_cache[model_key]

    def call(prompt: str) -> tuple[str, float]:
        categories, complexity = classify_task(prompt)
        primary_key = select_primary_model(categories, complexity)
        enhanced = build_specialist_prompt(
            prompt, categories, complexity, primary_key
        )

        t0 = time.perf_counter()

        try:
            raw = _get_llm(primary_key).invoke(enhanced).content.strip()
        except Exception as exc:
            try:
                raw = _get_llm(FALLBACK_MODEL).invoke(prompt).content.strip()
            except Exception as exc2:
                raw = f"[ERROR: {exc2}]"

        if should_synthesize(categories, complexity):
            try:
                syn_prompt = build_synthesis_prompt(
                    prompt, raw, categories, primary_key
                )
                raw = _get_llm(
                    "llama-8b-instant").invoke(syn_prompt).content.strip()
            except Exception:
                pass

        lat = (time.perf_counter() - t0) * 1000.0
        return raw, lat

    return call


# ══════════════════════════════════════════════════════════════════════════════
#  QA prompt
# ══════════════════════════════════════════════════════════════════════════════

def _qa_prompt(question: str, passages: list[str]) -> str:
    if passages and any(passages):
        context = "\n\n".join(
            f"[{i + 1}] {p}" for i, p in enumerate(passages[:4]) if p
        )
        if context:
            context_block = f"Passages:\n{context}\n\n"
        else:
            context_block = ""
    else:
        context_block = ""

    return (
        "Answer the question below as accurately as possible.\n"
        "Give a SHORT answer (a few words or one sentence). "
        "Do NOT explain or repeat the question.\n\n"
        + context_block
        + f"Question: {question}\n\n"
        "Short answer:"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════

def _normalise(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    for article in ("a", "an", "the"):
        text = re.sub(rf"\b{article}\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    return 1.0 if _normalise(pred) == _normalise(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    if not gold:
        return 0.0
    p_toks = _normalise(pred).split()
    g_toks = _normalise(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = sum(
        (collections.Counter(p_toks) & collections.Counter(g_toks)).values()
    )
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def retrieval_p_at_k(retrieved: list[str], gold: list[str], k: int = 4) -> float:
    if not gold or not retrieved:
        return 0.5
    gold_set = {t.lower() for t in gold if t}
    if not gold_set:
        return 0.5
    matches = sum(1 for t in retrieved[:k] if t.lower() in gold_set)
    return matches / k


def _simulate_retrieval(
    question: str,
    passages: list[str],
    titles: list[str],
    k: int = 4,
) -> tuple[list[str], list[str]]:
    """Rank passages by question-keyword overlap; return top-k titles & passages."""
    if not passages:
        return [], []
    q_tokens = set(_normalise(question).split())
    scores = []
    for t, p in zip(titles, passages):
        combined = f"{t} {p}" if t and p else (t or p or "")
        score = len(q_tokens & set(_normalise(combined).split()))
        scores.append(score)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return (
        [titles[i] for i in ranked[:k] if i < len(titles)],
        [passages[i] for i in ranked[:k] if i < len(passages)],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Single-model evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _evaluate(
    display_name: str,
    call_fn: Callable[[str], tuple[str, float]],
    dataset: list[dict],
    dataset_name: str,
) -> dict:
    em_list, f1_list, p4_list, lat_list = [], [], [], []
    errors = 0
    n = len(dataset)

    print(f"\n  ▶  {display_name}  ({dataset_name}, {n} questions)")

    for idx, item in enumerate(dataset):
        if (idx + 1) % 50 == 0:
            running_em = 100 * sum(em_list) / max(len(em_list), 1)
            print(f"     … {idx + 1}/{n}  (running EM: {running_em:.1f}%)")

        question = item["question"]
        gold_answer = item["answer"]
        titles = item.get("titles", [])
        passages = item.get("passages", [])
        gold_titles = item.get("gold_titles", [])

        top_titles, top_passages = _simulate_retrieval(
            question, passages, titles, k=4
        )

        prompt = _qa_prompt(question, top_passages)

        try:
            pred, lat = call_fn(prompt)
        except Exception as exc:
            print(f"     [ERROR] q{idx + 1}: {exc}")
            pred, lat = "", 0.0
            errors += 1

        pred = _strip_thinking(pred)
        pred = pred.split("\n")[0].strip()
        pred = re.sub(
            r"^(short answer[:\s]+|answer[:\s]+|the answer is[:\s]+)",
            "", pred, flags=re.IGNORECASE,
        ).strip()

        em_list.append(exact_match(pred, gold_answer))
        f1_list.append(token_f1(pred, gold_answer))
        p4_list.append(retrieval_p_at_k(top_titles, gold_titles, k=4))
        lat_list.append(lat)

    if errors:
        print(f"     [WARN] {errors}/{n} questions returned errors.")

    return {
        "em":  round(100 * sum(em_list) / max(len(em_list), 1), 1),
        "f1":  round(100 * sum(f1_list) / max(len(f1_list), 1), 1),
        "p4":  round(sum(p4_list) / max(len(p4_list), 1), 3),
        "lat": round(sum(lat_list) / max(len(lat_list), 1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Result table
# ══════════════════════════════════════════════════════════════════════════════

_COL = 26


# def _print_table(
#     dataset_name: str,
#     results: list[tuple[str, dict]],
#     n_questions: int,
# ) -> None:
#     print()
#     print("═" * 72)
#     print(f"  {dataset_name}   ({n_questions} questions)")
#     print("═" * 72)
#     print(
#         f"{'Model':<{_COL}}  {'EM (%)':<9} {'F1 (%)':<9} "
#         f"{'Retrieval P@4':<16} Lat (ms)"
#     )
#     print("─" * 72)
#     for name, m in results:
#         print(
#             f"{name:<{_COL}}  "
#             f"{m['em']:<9.1f} "
#             f"{m['f1']:<9.1f} "
#             f"{m['p4']:<16.3f} "
#             f"{m['lat']}"
#         )
#     print("═" * 72)
#     print()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     LLM Benchmark — Full Dataset Evaluation Only                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Max questions per file : {N_QUESTIONS}")
    print(f"  Questions directory    : {QUESTIONS_DIR}")
    print()

    # Discover files (auto-deletes samples and downloads full datasets)
    try:
        all_files = _discover_files()
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    # Filter to only full dataset files
    full_dataset_names = ["hotpotqa_full", "2wikimultihopqa_full"]
    selected_files = [f for f in all_files if f.stem in full_dataset_names]

    if not selected_files:
        print(
            f"\n[ERROR] No full datasets found. Expected: {full_dataset_names}")
        sys.exit(1)

    # Apply --datasets filter if given
    if args.datasets:
        filter_lower = {d.lower() for d in args.datasets}
        selected_files = [
            f for f in selected_files
            if f.stem.lower() in filter_lower
        ]
        if not selected_files:
            print(
                f"[ERROR] None of the requested datasets "
                f"({', '.join(args.datasets)}) matched files."
            )
            sys.exit(1)

    print(
        f"  Dataset files selected : {', '.join(p.name for p in selected_files)}")
    print()

    # Check Ollama
    needs_ollama = any(
        ollama_id not in ("adaptive",)
        for _, ollama_id, _ in MODELS
        if not any(s.lower() in _.lower() for s in args.skip)
        if not args.only or any(o.lower() in _.lower() for o in args.only)
    )
    if needs_ollama:
        _check_ollama()

    # Initialise model callers
    active: list[tuple[str, Callable]] = []

    for display_name, ollama_id, num_predict in MODELS:
        name_lower = display_name.lower()

        if args.only and not any(o.lower() in name_lower for o in args.only):
            continue
        if any(s.lower() in name_lower for s in args.skip):
            print(f"  [SKIP]   {display_name}")
            continue

        print(f"  [INIT]   {display_name:<30}  →  {ollama_id}")
        try:
            if ollama_id == "adaptive":
                fn = _build_adaptive_caller()
            else:
                fn = _build_ollama_caller(ollama_id, num_predict)

            active.append((display_name, fn))
            print("           OK")
        except Exception as exc:
            print(f"           FAILED — {exc}")

    if not active:
        print("\n[ERROR] No models could be initialised. Exiting.")
        sys.exit(1)

    print(f"\n  {len(active)} model(s) ready.\n")

    # Evaluate each dataset
    all_results: dict[str, list[tuple[str, dict]]] = {}

    for file_path in selected_files:
        dataset_name = file_path.stem

        print("─" * 60)
        print(f"  DATASET : {dataset_name.upper()}  ({file_path.name})")
        print("─" * 60)

        try:
            dataset = _load_file(file_path, N_QUESTIONS)
        except Exception as exc:
            print(f"  [ERROR] Could not load {file_path.name}: {exc}")
            continue

        print()

        dataset_results: list[tuple[str, dict]] = []
        for name, fn in active:
            m = _evaluate(name, fn, dataset, dataset_name)
            dataset_results.append((name, m))
            print(
                f"     → EM {m['em']:.1f}%  F1 {m['f1']:.1f}%  "
                f"P@4 {m['p4']:.3f}  Lat {m['lat']} ms"
            )

        all_results[dataset_name] = dataset_results

    # Print final results
    print("\n\n" + "═" * 72)
    print("  FINAL RESULTS")
    print("═" * 72)
    for ds_name, results in all_results.items():
        _print_table(ds_name, results, len(results[0][1]) if results else 0)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
