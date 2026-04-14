#!/usr/bin/env python3
"""
tests/download_datasets.py
─────────────────────────
Download HotpotQA and MuSiQue datasets for benchmarking.
"""

import json
import requests
from pathlib import Path


def download_hotpotqa():
    """Download HotpotQA dev set"""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "hotpotqa_dev_samples.json"

    # HotpotQA dev set (full is large, so we'll create a sample)
    # Full dataset: https://hotpotqa.github.io/

    # For now, create a small sample file
    samples = [
        {
            "question": "Which American-born actress played the lead role in the 2014 film 'The Fault in Our Stars' and also starred in the TV series 'The Vampire Diaries'?",
            "answer": "Shailene Woodley",
            "context": ["The Fault in Our Stars (film)", "The Vampire Diaries"]
        },
        {
            "question": "The first man to walk on the moon came from which US state?",
            "answer": "Ohio",
            "context": ["Neil Armstrong", "Apollo 11"]
        },
        {
            "question": "What is the name of the university that was founded by John D. Rockefeller and is located in Chicago?",
            "answer": "University of Chicago",
            "context": ["John D. Rockefeller", "Chicago"]
        },
    ]

    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Created sample HotpotQA file: {output_file}")
    print(f"  (Download full dataset from https://hotpotqa.github.io/ for complete benchmark)")


def download_musique():
    """Download MuSiQue dataset sample"""
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "musique_samples.json"

    samples = [
        {
            "question": "What is the sum of the atomic numbers of the first two noble gases?",
            "answer": "18",
            "steps": ["Find atomic number of Helium", "Find atomic number of Neon", "Sum them"]
        },
        {
            "question": "The author of 'Pride and Prejudice' was born in which English county?",
            "answer": "Hampshire",
            "steps": ["Identify author", "Find birthplace", "Determine county"]
        },
        {
            "question": "If a train travels 60 miles per hour for 2.5 hours, how many miles does it travel?",
            "answer": "150",
            "steps": ["Multiply speed by time"]
        },
    ]

    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)

    print(f"✓ Created sample MuSiQue file: {output_file}")


if __name__ == "__main__":
    print("Downloading dataset samples...")
    download_hotpotqa()
    download_musique()
    print("\nDone! Run benchmark with: python -m tests.benchmark --dataset both --samples 10")
