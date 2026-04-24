#!/usr/bin/env python3
"""
tests/complete_benchmark.py
──────────────────────────
Complete working benchmark for all models
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage

import argparse
import json
import time
import csv
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Only import what we need - avoid triggering the ModelProfile error

# Import chatbot - this will trigger the error if your graph has issues
# If you get ModelProfile error here, the issue is in your llms module
try:
    from app.core.graph import chatbot
    CHATBOT_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Could not import chatbot: {e}")
    print("⚠️  Make sure your app.core.graph doesn't have import issues")
    CHATBOT_AVAILABLE = False
    sys.exit(1)


@dataclass
class QueryResult:
    """Single query benchmark result"""
    query_id: int
    dataset: str
    model_key: str
    question: str
    expected_answer: str
    actual_answer: str
    is_correct: bool
    confidence: float
    latency_ms: float
    route_taken: str
    actual_model_used: str
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results"""
    dataset: str
    model_key: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    accuracy: float
    avg_confidence: float
    route_distribution: Dict[str, int] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# All available models (based on your table)
ALL_MODELS = [
    "deepseek_r1",
    "falcon3",
    "gemini_2_5_flash",
    "gemma3_270m",
    "granite3_dense_2b",
    "llama_3_1_8b_instant",
    "mistral_7b",
    "phi3_3_8b",
    "qwen2_5_coder_7b",
    "qwen3_5_0_8b"
]


def load_hotpotqa_samples(num_samples: int = 500) -> List[Dict]:
    """Load HotpotQA samples"""
    data_file = Path(__file__).parent / "data" / \
        f"hotpotqa_{num_samples}_samples.json"

    if not data_file.exists():
        print(f"⚠️  HotpotQA file not found: {data_file}")
        print("Creating sample data for testing...")
        samples = []
        base_questions = [
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "Who wrote Romeo and Juliet?",
                "answer": "William Shakespeare"},
            {"question": "What is the largest ocean on Earth?",
                "answer": "Pacific Ocean"},
            {"question": "Which planet is known as the Red Planet?", "answer": "Mars"},
            {"question": "What is the chemical symbol for gold?", "answer": "Au"},
            {"question": "Who painted the Mona Lisa?",
                "answer": "Leonardo da Vinci"},
            {"question": "What is the tallest mountain in the world?",
                "answer": "Mount Everest"},
            {"question": "Which country gifted the Statue of Liberty to the USA?",
                "answer": "France"},
            {"question": "What is the hardest natural substance?", "answer": "Diamond"},
            {"question": "Who developed the theory of relativity?",
                "answer": "Albert Einstein"},
        ]

        for i in range(min(num_samples, len(base_questions))):
            samples.append({
                "question": base_questions[i]["question"],
                "answer": base_questions[i]["answer"],
                "context": ["Sample context"],
                "type": "multi-hop"
            })
        return samples

    with open(data_file, 'r') as f:
        data = json.load(f)
    return data[:num_samples]


def load_musique_samples(num_samples: int = 500) -> List[Dict]:
    """Load MuSiQue samples"""
    data_file = Path(__file__).parent / "data" / \
        f"musique_{num_samples}_samples.json"

    if not data_file.exists():
        print(f"⚠️  MuSiQue file not found: {data_file}")
        print("Creating sample data for testing...")
        samples = []
        base_questions = [
            {"question": "What is 15% of 200?", "answer": "30"},
            {"question": "If a train travels 60 mph for 2.5 hours, how far does it go?", "answer": "150"},
            {"question": "What is the square root of 144?", "answer": "12"},
            {"question": "Calculate (45 + 27) * 3", "answer": "216"},
            {"question": "What is 100 divided by 4?", "answer": "25"},
            {"question": "What is the atomic number of Hydrogen?", "answer": "1"},
            {"question": "Who wrote 'Pride and Prejudice'?", "answer": "Jane Austen"},
            {"question": "What is the boiling point of water in Celsius?", "answer": "100"},
        ]

        for i in range(min(num_samples, len(base_questions))):
            samples.append({
                "question": base_questions[i]["question"],
                "answer": base_questions[i]["answer"],
                "steps": ["Reasoning step"],
                "type": "multi-hop"
            })
        return samples

    with open(data_file, 'r') as f:
        data = json.load(f)
    return data[:num_samples]


def exact_match_eval(expected: str, actual: str) -> Tuple[bool, float]:
    """Evaluate answer using exact match"""
    if not expected or not actual:
        return False, 0.0

    expected_clean = expected.lower().strip()
    actual_clean = actual.lower().strip()

    # Remove punctuation
    import string
    expected_clean = expected_clean.translate(
        str.maketrans('', '', string.punctuation))
    actual_clean = actual_clean.translate(
        str.maketrans('', '', string.punctuation))

    if expected_clean == actual_clean:
        return True, 1.0

    if expected_clean in actual_clean:
        return True, 0.9

    if actual_clean in expected_clean:
        return True, 0.8

    # Check for numerical equivalence
    try:
        expected_num = float(expected_clean)
        actual_num = float(actual_clean)
        if abs(expected_num - actual_num) < 0.01:
            return True, 0.95
    except ValueError:
        pass

    return False, 0.0


def run_query_with_model(question: str, model_key: str) -> Tuple[Optional[str], float, Optional[str], str, str]:
    """
    Run a single query with a specific model
    """
    start_time = time.time()
    answer = None
    error = None
    route_taken = "unknown"
    actual_model_used = model_key

    try:
        unique_thread_id = f"bench_{model_key}_{int(start_time)}_{hash(question) % 10000}"

        inputs = {
            "messages": [HumanMessage(content=question)],
            "model": model_key,
            "routing_info": None,
            "rag_context": None,
        }

        config = {
            "configurable": {
                "thread_id": unique_thread_id
            }
        }

        result = chatbot.invoke(inputs, config=config)

        if isinstance(result, dict):
            messages = result.get("messages", [])
            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    answer = last_msg.content
                elif isinstance(last_msg, dict):
                    answer = last_msg.get("content", "")

            route_taken = result.get("route", "unknown")
            routing_info = result.get("routing_info", {})
            if routing_info:
                actual_model_used = routing_info.get("model_key", model_key)

            # Remove attribution footer for evaluation
            if answer and "---" in answer:
                answer = answer.split("---")[0].strip()

        if not answer:
            error = "No answer returned from graph"

    except Exception as e:
        error = str(e)
        answer = None

    latency_ms = (time.time() - start_time) * 1000
    return answer, latency_ms, error, route_taken, actual_model_used


def benchmark_model_on_dataset(
    model_key: str,
    samples: List[Dict],
    dataset_name: str,
    verbose: bool = True
) -> Tuple[List[QueryResult], BenchmarkSummary]:
    """Benchmark a single model on a dataset"""

    results = []
    correct_count = 0
    route_counts = {}
    total_confidence = 0.0

    print(f"\n{'='*70}")
    print(f"📊 Benchmarking: {model_key} on {dataset_name.upper()}")
    print(f"📝 Total samples: {len(samples)}")
    print(f"{'='*70}\n")

    for idx, sample in enumerate(samples):
        question = sample["question"]
        expected = sample.get("answer", "")

        if verbose:
            print(f"[{idx+1}/{len(samples)}] Model: {model_key}")
            print(f"   Q: {question[:80]}...")

        answer, latency_ms, error, route_taken, actual_model = run_query_with_model(
            question, model_key)

        is_correct = False
        confidence = 0.0

        if answer and not error:
            is_correct, confidence = exact_match_eval(expected, answer)
            if is_correct:
                correct_count += 1
            total_confidence += confidence

            if verbose:
                print(f"   ✓ Answer: {answer[:80]}...")
                print(
                    f"   ✓ Correct: {is_correct} (confidence: {confidence:.2f})")
                print(
                    f"   ✓ Route: {route_taken} | Actual model: {actual_model}")
        else:
            if verbose:
                print(f"   ✗ Error: {error}")

        route_counts[route_taken] = route_counts.get(route_taken, 0) + 1

        result = QueryResult(
            query_id=idx,
            dataset=dataset_name,
            model_key=model_key,
            question=question,
            expected_answer=expected,
            actual_answer=answer or "[ERROR]",
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            route_taken=route_taken,
            actual_model_used=actual_model,
            error=error
        )

        results.append(result)

        if verbose:
            print(f"   ⏱️  Latency: {latency_ms:.0f}ms\n")

        time.sleep(0.5)

    latencies = [r.latency_ms for r in results if not r.error]
    latencies.sort()

    successful = len([r for r in results if not r.error])

    summary = BenchmarkSummary(
        dataset=dataset_name,
        model_key=model_key,
        total_queries=len(results),
        successful_queries=successful,
        failed_queries=len([r for r in results if r.error]),
        avg_latency_ms=sum(latencies)/len(latencies) if latencies else 0,
        median_latency_ms=latencies[len(latencies)//2] if latencies else 0,
        p95_latency_ms=latencies[int(len(latencies)*0.95)] if len(
            latencies) >= 20 else (latencies[-1] if latencies else 0),
        min_latency_ms=latencies[0] if latencies else 0,
        max_latency_ms=latencies[-1] if latencies else 0,
        accuracy=correct_count / len(results) if results else 0,
        avg_confidence=(total_confidence /
                        successful) if successful > 0 else 0,
        route_distribution=route_counts
    )

    return results, summary


def save_results(results: List[QueryResult], summary: BenchmarkSummary, output_dir: Path):
    """Save benchmark results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_file = output_dir / \
        f"{summary.dataset}_{summary.model_key}_{timestamp}_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(asdict(summary), f, indent=2)

    results_file = output_dir / \
        f"{summary.dataset}_{summary.model_key}_{timestamp}_results.json"
    with open(results_file, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_file = output_dir / \
        f"{summary.dataset}_{summary.model_key}_{timestamp}_results.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = list(asdict(results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                writer.writerow(row)

    print(f"💾 Saved results to: {output_dir}")


def print_summary_table(results_dict: Dict[str, BenchmarkSummary], dataset_name: str):
    """Print formatted summary table"""

    print(f"\n{'='*90}")
    print(f"📊 {dataset_name.upper()} RESULTS SUMMARY")
    print(f"{'='*90}")
    print(f"{'Model':<30} {'Accuracy':<12} {'Avg Latency':<15} {'Success Rate':<15} {'Route Dist':<20}")
    print(f"{'-'*90}")

    for model_key, summary in sorted(results_dict.items(), key=lambda x: -x[1].accuracy):
        route_str = ", ".join(
            [f"{r}:{c}" for r, c in summary.route_distribution.items()])
        print(f"{model_key:<30} {summary.accuracy*100:>6.1f}%      {summary.avg_latency_ms:>8.0f}ms        {summary.successful_queries/summary.total_queries*100:>6.1f}%        {route_str}")

    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Complete model benchmark")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of samples per dataset")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS[:3],
                        help=f"Models to test. Available: {ALL_MODELS}")
    parser.add_argument("--dataset", type=str, default="both",
                        choices=["hotpotqa", "musique", "both"])
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")

    args = parser.parse_args()

    if args.quick:
        args.samples = 1
        print("⚡ QUICK MODE: Testing with 1 sample per model")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print(f"\n📚 Loading datasets...")
    hotpotqa_samples = load_hotpotqa_samples(args.samples) if args.dataset in [
        "hotpotqa", "both"] else []
    musique_samples = load_musique_samples(args.samples) if args.dataset in [
        "musique", "both"] else []

    print(f"✓ Loaded {len(hotpotqa_samples)} HotpotQA samples")
    print(f"✓ Loaded {len(musique_samples)} MuSiQue samples")

    hotpot_results_dict = {}
    musique_results_dict = {}

    for model_key in args.models:
        print(f"\n{'🔥'*40}")
        print(f"Testing model: {model_key}")
        print(f"{'🔥'*40}")

        if hotpotqa_samples:
            print(f"\n📖 Running HotpotQA benchmark...")
            try:
                hotpot_results, hotpot_summary = benchmark_model_on_dataset(
                    model_key, hotpotqa_samples, "hotpotqa", verbose=not args.quick
                )
                model_dir = output_dir / model_key
                save_results(hotpot_results, hotpot_summary,
                             model_dir / "hotpotqa")
                hotpot_results_dict[model_key] = hotpot_summary
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

        if musique_samples:
            print(f"\n📖 Running MuSiQue benchmark...")
            try:
                musique_results, musique_summary = benchmark_model_on_dataset(
                    model_key, musique_samples, "musique", verbose=not args.quick
                )
                model_dir = output_dir / model_key
                save_results(musique_results, musique_summary,
                             model_dir / "musique")
                musique_results_dict[model_key] = musique_summary
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()

    if hotpot_results_dict:
        print_summary_table(hotpot_results_dict, "HotpotQA")
    if musique_results_dict:
        print_summary_table(musique_results_dict, "MuSiQue")

    print(f"\n✅ Benchmark complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
