#!/usr/bin/env python3
"""
tests/benchmark.py
──────────────────
Benchmark script for HotpotQA and MuSiQue datasets.
"""

from __future__ import annotations
from app.models.chat_state import ChatState
from app.core.graph import chatbot
from langchain_core.messages import HumanMessage

import argparse
import json
import time
import csv
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# CRITICAL: Add project root to path correctly
# Get the absolute path to the project root (parent of tests directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Change working directory to project root
os.chdir(PROJECT_ROOT)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python path: {sys.path[0]}")
print(f"Working directory: {os.getcwd()}")

# Now import


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Single query benchmark result"""
    query_id: int
    dataset: str
    question: str
    expected_answer: str
    actual_answer: str
    route_taken: str
    model_used: str
    is_correct: bool
    confidence: float
    latency_ms: float
    estimated_tokens: int
    error: Optional[str] = None
    routing_reason: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    complexity: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results"""
    dataset: str
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    accuracy: float
    route_distribution: Dict[str, int]
    route_accuracy: Dict[str, float]
    route_avg_latency: Dict[str, float]
    model_distribution: Dict[str, int]
    model_accuracy: Dict[str, float]
    category_distribution: Dict[str, int]
    category_accuracy: Dict[str, float]
    complexity_distribution: Dict[str, int]
    complexity_accuracy: Dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ── Dataset Loaders ──────────────────────────────────────────────────────────

def load_custom_test_samples() -> List[Dict]:
    """Load custom test samples covering all routing paths."""
    return [
        # Direct path (no retrieval needed)
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "type": "direct",
            "dataset": "custom"
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare",
            "type": "direct",
            "dataset": "custom"
        },
        {
            "question": "What is the chemical symbol for gold?",
            "answer": "Au",
            "type": "direct",
            "dataset": "custom"
        },
        {
            "question": "What is the largest ocean on Earth?",
            "answer": "Pacific Ocean",
            "type": "direct",
            "dataset": "custom"
        },

        # Agent path (needs calculation or real-time info)
        {
            "question": "What is 15% of 200?",
            "answer": "30",
            "type": "calculation",
            "dataset": "custom"
        },
        {
            "question": "Calculate (45 + 27) * 3",
            "answer": "216",
            "type": "calculation",
            "dataset": "custom"
        },
        {
            "question": "What is 100 divided by 4?",
            "answer": "25",
            "type": "calculation",
            "dataset": "custom"
        },

        # RAG path (needs document retrieval - will fallback to direct if no docs)
        {
            "question": "According to the uploaded document, what is the main topic?",
            "answer": "",  # Depends on uploaded documents
            "type": "rag",
            "dataset": "custom"
        },
        {
            "question": "Summarize the key points from the document",
            "answer": "",
            "type": "rag",
            "dataset": "custom"
        },
    ]


# ── Answer Evaluation ─────────────────────────────────────────────────────────

def exact_match_eval(expected: str, actual: str) -> Tuple[bool, float]:
    """Evaluate answer using exact match (case-insensitive, stripped)."""
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


def evaluate_answer(question: str, expected: str, actual: str) -> Tuple[bool, float]:
    """Evaluate answer correctness."""
    if not expected:
        # No expected answer (e.g., real-time query)
        return bool(actual and len(actual) > 10), 0.5

    return exact_match_eval(expected, actual)


# ── Query Execution ───────────────────────────────────────────────────────────

def extract_routing_info(result: dict) -> Dict:
    """Extract routing information from result."""
    routing_info = result.get("routing_info", {})

    return {
        "route": result.get("route", "unknown"),
        "model_used": routing_info.get("model_key", "unknown") if routing_info else "unknown",
        "model_name": routing_info.get("model_name", "unknown") if routing_info else "unknown",
        "reason": routing_info.get("reason", "") if routing_info else "",
        "categories": routing_info.get("categories", []) if routing_info else [],
        "complexity": routing_info.get("complexity", "unknown") if routing_info else "unknown",
        "auto_selected": routing_info.get("auto_selected", False) if routing_info else False,
    }


def run_single_query(
    question: str,
    thread_id: str = "benchmark"
) -> Tuple[Optional[str], Dict, float, Optional[str]]:
    """
    Run a single query through the graph.

    Returns:
        tuple: (answer, routing_info, latency_ms, error)
    """
    start_time = time.time()
    answer = None
    routing_info = {}
    error = None

    try:
        # Prepare input as ChatState
        from langchain_core.messages import HumanMessage

        # Create a unique thread ID for this query
        unique_thread_id = f"{thread_id}_{int(start_time)}_{hash(question) % 10000}"

        # Prepare the state
        inputs = {
            "messages": [HumanMessage(content=question)],
            "model": "llama-8b-instant",  # Default model
            "routing_info": None,
            "rag_context": None,
        }

        config = {
            "configurable": {
                "thread_id": unique_thread_id
            }
        }

        print(f"  [DEBUG] Running query: {question[:50]}...")
        print(f"  [DEBUG] Thread ID: {unique_thread_id}")

        # Run through graph
        result = chatbot.invoke(inputs, config=config)

        print(f"  [DEBUG] Result type: {type(result)}")
        print(
            f"  [DEBUG] Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")

        # Extract answer from result
        if isinstance(result, dict):
            messages = result.get("messages", [])
            print(f"  [DEBUG] Messages count: {len(messages)}")

            if messages:
                last_msg = messages[-1]
                if hasattr(last_msg, 'content'):
                    answer = last_msg.content
                    print(
                        f"  [DEBUG] Answer length: {len(answer) if answer else 0}")
                elif isinstance(last_msg, dict):
                    answer = last_msg.get("content", "")
                    print(
                        f"  [DEBUG] Answer (dict): {answer[:100] if answer else 'None'}")

            # Extract routing info
            routing_info = extract_routing_info(result)
            print(f"  [DEBUG] Route: {routing_info.get('route', 'unknown')}")
            print(
                f"  [DEBUG] Model: {routing_info.get('model_used', 'unknown')}")

        latency_ms = (time.time() - start_time) * 1000

        if not answer:
            error = "No answer returned from graph"
            print(f"  [DEBUG] ERROR: {error}")

        return answer, routing_info, latency_ms, error

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        error = str(e)
        print(f"  [DEBUG] EXCEPTION: {error}")
        import traceback
        traceback.print_exc()
        return None, {}, latency_ms, error


def estimate_tokens(text: str) -> int:
    """Rough token estimation."""
    if not text:
        return 0
    return len(text) // 4


# ── Benchmark Runner ─────────────────────────────────────────────────────────

def run_benchmark(
    samples: List[Dict],
    dataset_name: str,
    verbose: bool = True
) -> Tuple[List[QueryResult], BenchmarkSummary]:
    """Run benchmark on a list of samples."""
    results: List[QueryResult] = []

    print(f"\n{'='*60}")
    print(f"Running Benchmark: {dataset_name}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*60}\n")

    for idx, sample in enumerate(samples):
        question = sample["question"]
        expected = sample.get("answer", "")

        print(f"[{idx+1}/{len(samples)}] Question: {question[:80]}...")

        # Run query
        answer, routing_info, latency_ms, error = run_single_query(question)

        # Evaluate correctness
        is_correct = False
        confidence = 0.0

        if answer and not error:
            is_correct, confidence = evaluate_answer(
                question, expected, answer)
            print(f"  ✓ Answer received ({len(answer)} chars)")
            print(f"  ✓ Correct: {is_correct}, Confidence: {confidence}")
        else:
            print(f"  ✗ Failed: {error or 'No answer'}")

        # Estimate tokens
        estimated_tokens = estimate_tokens(
            question) + estimate_tokens(answer or "")

        # Create result
        result = QueryResult(
            query_id=idx,
            dataset=dataset_name,
            question=question,
            expected_answer=expected,
            actual_answer=answer or "[ERROR]",
            route_taken=routing_info.get("route", "unknown"),
            model_used=routing_info.get("model_used", "unknown"),
            is_correct=is_correct,
            confidence=confidence,
            latency_ms=latency_ms,
            estimated_tokens=estimated_tokens,
            error=error,
            routing_reason=routing_info.get("reason", ""),
            categories=routing_info.get("categories", []),
            complexity=routing_info.get("complexity", "unknown"),
        )

        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"  {status} Route: {result.route_taken} | Model: {result.model_used} | "
              f"Latency: {latency_ms:.0f}ms\n")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Generate summary
    summary = generate_summary(results, dataset_name)

    return results, summary


def generate_summary(results: List[QueryResult], dataset_name: str) -> BenchmarkSummary:
    """Generate aggregated summary from results."""
    total = len(results)
    successful = sum(1 for r in results if not r.error)
    failed = total - successful

    # Latency statistics
    latencies = [
        r.latency_ms for r in results if not r.error and r.latency_ms > 0]
    if latencies:
        latencies.sort()
        avg_latency = sum(latencies) / len(latencies)
        median_latency = latencies[len(latencies) // 2]
        p95_latency = latencies[int(
            len(latencies) * 0.95)] if len(latencies) >= 20 else max(latencies)
    else:
        avg_latency = median_latency = p95_latency = 0

    # Overall accuracy
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total if total > 0 else 0

    # Route breakdown
    route_distribution = {}
    route_correct = {}
    route_latencies = {}

    for r in results:
        route = r.route_taken
        route_distribution[route] = route_distribution.get(route, 0) + 1
        if r.is_correct:
            route_correct[route] = route_correct.get(route, 0) + 1
        route_latencies[route] = route_latencies.get(route, 0) + r.latency_ms

    route_accuracy = {
        route: route_correct.get(route, 0) / route_distribution[route]
        for route in route_distribution
    }
    route_avg_latency = {
        route: route_latencies[route] / route_distribution[route]
        for route in route_distribution
    }

    # Model breakdown
    model_distribution = {}
    model_correct = {}

    for r in results:
        model = r.model_used
        model_distribution[model] = model_distribution.get(model, 0) + 1
        if r.is_correct:
            model_correct[model] = model_correct.get(model, 0) + 1

    model_accuracy = {
        model: model_correct.get(model, 0) / model_distribution[model]
        for model in model_distribution
    }

    # Category breakdown
    category_distribution = {}
    category_correct = {}

    for r in results:
        for cat in r.categories:
            category_distribution[cat] = category_distribution.get(cat, 0) + 1
            if r.is_correct:
                category_correct[cat] = category_correct.get(cat, 0) + 1

    category_accuracy = {
        cat: category_correct.get(cat, 0) / category_distribution[cat]
        for cat in category_distribution
    }

    # Complexity breakdown
    complexity_distribution = {}
    complexity_correct = {}

    for r in results:
        comp = r.complexity
        complexity_distribution[comp] = complexity_distribution.get(
            comp, 0) + 1
        if r.is_correct:
            complexity_correct[comp] = complexity_correct.get(comp, 0) + 1

    complexity_accuracy = {
        comp: complexity_correct.get(comp, 0) / complexity_distribution[comp]
        for comp in complexity_distribution
    }

    return BenchmarkSummary(
        dataset=dataset_name,
        total_queries=total,
        successful_queries=successful,
        failed_queries=failed,
        avg_latency_ms=avg_latency,
        median_latency_ms=median_latency,
        p95_latency_ms=p95_latency,
        accuracy=accuracy,
        route_distribution=route_distribution,
        route_accuracy=route_accuracy,
        route_avg_latency=route_avg_latency,
        model_distribution=model_distribution,
        model_accuracy=model_accuracy,
        category_distribution=category_distribution,
        category_accuracy=category_accuracy,
        complexity_distribution=complexity_distribution,
        complexity_accuracy=complexity_accuracy,
    )


# ── Output Saving ────────────────────────────────────────────────────────────

def save_results(
    results: List[QueryResult],
    summary: BenchmarkSummary,
    output_dir: Path = None
):
    """Save benchmark results to JSON and CSV files."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "results"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset = summary.dataset

    # Save detailed results as JSON
    json_path = output_dir / f"{dataset}_{timestamp}_results.json"
    with open(json_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n📁 Detailed results saved to: {json_path}")

    # Save summary as JSON
    summary_path = output_dir / f"{dataset}_{timestamp}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"📁 Summary saved to: {summary_path}")

    # Save as CSV
    csv_path = output_dir / f"{dataset}_{timestamp}_results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            fieldnames = list(asdict(results[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                row = asdict(r)
                row['categories'] = ','.join(row['categories'])
                writer.writerow(row)
    print(f"📁 CSV saved to: {csv_path}")

    return json_path, csv_path


def print_summary(summary: BenchmarkSummary):
    """Pretty print benchmark summary."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY: {summary.dataset}")
    print(f"{'='*60}")

    print(f"\n📊 Overall Statistics:")
    print(f"   Total Queries:    {summary.total_queries}")
    print(f"   Successful:       {summary.successful_queries}")
    print(f"   Failed:           {summary.failed_queries}")
    print(f"   Accuracy:         {summary.accuracy*100:.1f}%")
    print(f"   Avg Latency:      {summary.avg_latency_ms:.0f}ms")
    print(f"   Median Latency:   {summary.median_latency_ms:.0f}ms")
    print(f"   P95 Latency:      {summary.p95_latency_ms:.0f}ms")

    print(f"\n🎯 Route Distribution & Accuracy:")
    for route, count in sorted(summary.route_distribution.items()):
        acc = summary.route_accuracy.get(route, 0) * 100
        lat = summary.route_avg_latency.get(route, 0)
        print(
            f"   {route:8s}: {count:3d} queries | Accuracy: {acc:5.1f}% | Avg Latency: {lat:.0f}ms")

    print(f"\n🤖 Model Distribution & Accuracy:")
    for model, count in sorted(summary.model_distribution.items(), key=lambda x: -x[1]):
        acc = summary.model_accuracy.get(model, 0) * 100
        print(f"   {model:20s}: {count:3d} queries | Accuracy: {acc:5.1f}%")

    print(f"\n{'='*60}\n")


# ── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="custom",
        choices=["custom"],
        help="Dataset to benchmark"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Verify chatbot loaded
    print(f"Chatbot loaded: {chatbot is not None}")
    print(f"Chatbot type: {type(chatbot)}")

    # Run benchmark
    samples = load_custom_test_samples()
    if args.samples < len(samples):
        samples = samples[:args.samples]

    results, summary = run_benchmark(samples, "custom", verbose)
    save_results(results, summary)
    print_summary(summary)


if __name__ == "__main__":
    main()
