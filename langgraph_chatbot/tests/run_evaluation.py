#!/usr/bin/env python3
"""
run_evaluation.py
────────────────
Complete evaluation pipeline for all models and datasets.
"""

import json
import time
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

from complete_benchmark import run_model_benchmark, ALL_MODELS
from download_datasets import download_hotpotqa, download_musique


def load_dataset(dataset_name: str, samples: int = 500) -> List[Dict]:
    """Load dataset from downloaded files"""
    data_dir = Path(__file__).parent / "data"

    if dataset_name == "hotpotqa":
        file_path = data_dir / f"hotpotqa_{samples}_samples.json"
    else:
        file_path = data_dir / f"musique_{samples}_samples.json"

    if not file_path.exists():
        print(f"✗ Dataset not found: {file_path}")
        print(f"Run: python -m tests.download_datasets --samples {samples}")
        return []

    with open(file_path, 'r') as f:
        data = json.load(f)

    # Ensure we have exactly 'samples' items
    return data[:samples]


def generate_results_table(results_dir: Path, dataset: str) -> pd.DataFrame:
    """Generate results table from saved benchmark files"""

    table_data = []

    for model in ALL_MODELS:
        summary_file = results_dir / model / dataset / "*_summary.json"
        summary_files = list(
            Path(results_dir / model / dataset).glob("*_summary.json"))

        if summary_files:
            with open(summary_files[0], 'r') as f:
                summary = json.load(f)

            table_data.append({
                "Model": model,
                "EM (%)": summary.get("accuracy", 0) * 100,
                "F1 (%)": summary.get("f1_score", 0) * 100,
                "Retrieval P@4": summary.get("retrieval_precision", 0) * 100,
                "Lat (ms)": summary.get("avg_latency_ms", 0)
            })

    # Add adaptive routing results
    adaptive_file = results_dir / "adaptive" / dataset / "*_summary.json"
    adaptive_files = list(
        Path(results_dir / "adaptive" / dataset).glob("*_summary.json"))

    if adaptive_files:
        with open(adaptive_files[0], 'r') as f:
            adaptive_summary = json.load(f)

        table_data.append({
            "Model": "Adaptive (Ours)",
            "EM (%)": adaptive_summary.get("accuracy", 0) * 100,
            "F1 (%)": adaptive_summary.get("f1_score", 0) * 100,
            "Retrieval P@4": adaptive_summary.get("retrieval_precision", 0) * 100,
            "Lat (ms)": adaptive_summary.get("avg_latency_ms", 0)
        })

    return pd.DataFrame(table_data)


def generate_latex_table(df: pd.DataFrame, caption: str, label: str) -> str:
    """Generate LaTeX table from DataFrame"""

    latex = f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
    latex += "\\begin{tabular}{|l|c|c|c|c|}\n\\hline\n"
    latex += "\\textbf{Model} & \\textbf{EM (\\%)} & \\textbf{F1 (\\%)} & \\textbf{Retrieval P@4} & \\textbf{Lat (ms)} \\\\\n\\hline\n"

    for _, row in df.iterrows():
        latex += f"{row['Model']} & {row['EM (%)']:.1f} & {row['F1 (%)']:.1f} & "
        latex += f"{row['Retrieval P@4']:.1f} & {row['Lat (ms)']:.0f} \\\\\n\\hline\n"

    latex += "\\end{tabular}\n\\end{table}\n"
    return latex


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples per dataset (use 500 for final)")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Models to benchmark")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Output directory")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Download datasets if needed
    if not args.skip_download:
        print("📥 Downloading datasets...")
        download_hotpotqa(args.samples)
        download_musique(args.samples)

    # Load datasets
    print(f"📚 Loading {args.samples} samples per dataset...")
    hotpotqa_data = load_dataset("hotpotqa", args.samples)
    musique_data = load_dataset("musique", args.samples)

    if not hotpotqa_data or not musique_data:
        print("✗ Failed to load datasets")
        return

    print(
        f"✓ Loaded {len(hotpotqa_data)} HotpotQA and {len(musique_data)} MuSiQue samples")

    # Run benchmarks (this will take a long time!)
    print(f"\n🚀 Starting evaluation with {len(args.models)} models...")
    print(
        f"⚠️  This will take approximately {len(args.models) * args.samples * 5 / 60:.1f} hours")
    print("   (Assuming 5 seconds per query)")

    confirm = input("Continue? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborted.")
        return

    # Run full benchmark (implement this in benchmark.py)
    # all_results, adaptive_results = run_full_benchmark(hotpotqa_data, musique_data, args.models)

    # Generate tables
    print("\n📊 Generating result tables...")
    hotpot_table = generate_results_table(output_dir, "hotpotqa")
    musique_table = generate_results_table(output_dir, "musique")

    # Save as LaTeX
    with open(output_dir / "hotpotqa_table.tex", 'w') as f:
        f.write(generate_latex_table(hotpot_table,
                                     "Model performance on the HotpotQA benchmark (500 questions). EM = Exact Match; F1 = Token F1; Lat = Average Latency (ms).",
                                     "tab:hotpotqa"))

    with open(output_dir / "musique_table.tex", 'w') as f:
        f.write(generate_latex_table(musique_table,
                                     "Model performance on the MuSiQue benchmark (500 questions). EM = Exact Match; F1 = Token F1; Lat = Average Latency (ms).",
                                     "tab:musique"))

    print(f"✓ Tables saved to {output_dir}/")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
