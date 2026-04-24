#!/usr/bin/env python3
"""
tests/create_sample_data.py
──────────────────────────
Create sample datasets for testing when downloads fail
"""

import json
from pathlib import Path


def create_sample_hotpotqa(num_samples: int = 500):
    """Create sample HotpotQA data"""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    samples = []
    base_questions = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Romeo and Juliet?",
            "answer": "William Shakespeare"},
        {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
        {"question": "Which planet is known as the Red Planet?", "answer": "Mars"},
        {"question": "What is the chemical symbol for gold?", "answer": "Au"},
        {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
        {"question": "What is the tallest mountain in the world?",
            "answer": "Mount Everest"},
        {"question": "Which country gifted the Statue of Liberty to the USA?",
            "answer": "France"},
        {"question": "What is the hardest natural substance?", "answer": "Diamond"},
        {"question": "Who developed the theory of relativity?",
            "answer": "Albert Einstein"},
    ]

    for i in range(num_samples):
        sample = base_questions[i % len(base_questions)].copy()
        sample["context"] = ["Sample context for HotpotQA"]
        sample["type"] = "multi-hop"
        samples.append(sample)

    output_file = output_dir / f"hotpotqa_{num_samples}_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Created {num_samples} HotpotQA samples in {output_file}")


def create_sample_musique(num_samples: int = 500):
    """Create sample MuSiQue data"""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    samples = []
    base_questions = [
        {"question": "What is 15% of 200?", "answer": "30"},
        {"question": "If a train travels 60 mph for 2.5 hours, how far does it go?", "answer": "150"},
        {"question": "What is the square root of 144?", "answer": "12"},
        {"question": "Calculate (45 + 27) * 3", "answer": "216"},
        {"question": "What is 100 divided by 4?", "answer": "25"},
    ]

    for i in range(num_samples):
        sample = base_questions[i % len(base_questions)].copy()
        sample["steps"] = ["Calculation step"]
        sample["type"] = "multi-hop"
        samples.append(sample)

    output_file = output_dir / f"musique_{num_samples}_samples.json"
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Created {num_samples} MuSiQue samples in {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    create_sample_hotpotqa(args.samples)
    create_sample_musique(args.samples)
    print("\n✅ Sample data created! Run benchmark with:")
    print(f"   python -m tests.complete_benchmark --samples {args.samples}")
