"""
evaluate_benchmarks.py
──────────────────────
Unified benchmark evaluation loading questions from a local
`testing_questions/` folder (JSON or TXT files).

Evaluates all models through Ollama (local) including the Adaptive (Ours)
custom orchestrator.

Metrics:
  EM   — Exact Match (%)
  F1   — Token-level F1 (%)
  P@4  — Retrieval Precision@4 (keyword-overlap simulated retrieval)
  Lat  — Mean latency per question (ms)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  testing_questions/ — supported file formats
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  JSON (.json)
  ────────────
  A JSON array of objects.  Only `question` is required; every other
  field has a sensible default (empty string / empty list).

    [
      {
        "question":    "Which city hosted the 2012 Olympics?",
        "answer":      "London",
        "passages":    ["London hosted the 2012 Summer Olympics …"],
        "titles":      ["2012 Summer Olympics"],
        "gold_titles": ["2012 Summer Olympics"]
      },
      …
    ]

  TXT (.txt)
  ──────────
  One entry per line.  Three micro-formats are accepted:

    • Question only (no gold answer — EM/F1 will score 0):
        Which city hosted the 2012 Olympics?

    • Tab-separated  question <TAB> answer:
        Which city hosted the 2012 Olympics?	London

    • Pipe-separated  question | answer:
        Which city hosted the 2012 Olympics? | London

  Blank lines and lines starting with '#' are ignored.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prerequisites
─────────────
1.  Ollama must be running:
        ollama serve

2.  Pull every model once:
        ollama pull deepseek-r1:latest
        ollama pull falcon3:latest
        ollama pull gemma3:270m
        ollama pull granite3-dense:2b
        ollama pull llama3.1:8b
        ollama pull mistral:7b
        ollama pull phi3:3.8b
        ollama pull qwen2.5-coder:7b
        ollama pull qwen3.5:0.8b

3.  Install extra dependencies if needed:
        pip install langchain-ollama datasets

4.  Place this file in your project root alongside the `llms/` package and the
    `testing_questions/` folder.

Usage
─────
    # Full run — all files in testing_questions/:
    python evaluate_benchmarks.py

    # Limit to 50 questions per file (smoke-test):
    python evaluate_benchmarks.py --n 50

    # Only specific dataset files (match by filename stem, case-insensitive):
    python evaluate_benchmarks.py --datasets hotpotqa my_custom_set

    # Skip slow / unavailable models:
    python evaluate_benchmarks.py --skip deepseek-r1 adaptive

    # Run a specific subset of models:
    python evaluate_benchmarks.py --only mistral adaptive
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path
from typing import Callable

# ── CLI arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Benchmark all LLMs using local testing_questions/ folder"
)
parser.add_argument(
    "--datasets", nargs="*", default=None,
    help=(
        "Filename stems (no extension) to evaluate, e.g. 'hotpotqa my_set'. "
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

N_QUESTIONS: int = args.n

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
QUESTIONS_DIR = PROJECT_ROOT / "testing_questions"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Model table ────────────────────────────────────────────────────────────────
#
# Columns:
#   display_name  — label printed in results
#   ollama_id     — Ollama tag, OR "adaptive" (orchestrator)
#   num_predict   — max tokens (ignored for adaptive)
#
MODELS: list[tuple[str, str, int]] = [
    # (display_name,              ollama_id,                      num_predict)
    ("DeepSeek-R1",              "deepseek-r1:latest",           256),
    ("Falcon3",                  "falcon3:latest",               256),
    ("Gemma3-270M",              "gemma3:270m",                  128),
    ("Granite3-Dense-2B",        "granite3-dense:2b",            256),
    ("LLaMA-3.1-8B-Instant",     "llama3.1:8b",                  256),
    ("Mistral-7B",               "mistral:7b",                   256),
    ("Phi3-3.8B",                "phi3:3.8b",                    256),
    ("Qwen2.5-Coder-7B",         "qwen2.5-coder:7b",             256),
    ("Qwen3.5-0.8B",             "qwen3.5:0.8b",                 128),
    ("Adaptive (Ours)",          "adaptive",                     256),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset loading — testing_questions/ folder
# ══════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
#  Built-in benchmark questions
#  These are written to testing_questions/ automatically when the folder is
#  empty, so no manual setup is ever required.
# ---------------------------------------------------------------------------

_HOTPOTQA_QUESTIONS: list[dict] = [
    {"question": "Were Scott Derrickson and Ed Wood of the same nationality?",
     "answer": "yes",
     "passages": [
         "Scott Derrickson is an American director, screenwriter and producer.",
         "Ed Wood was an American filmmaker, actor, writer, producer, and director.",
     ],
     "titles": ["Scott Derrickson", "Ed Wood"],
     "gold_titles": ["Scott Derrickson", "Ed Wood"]},

    {"question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
     "answer": "Chief of Protocol",
     "passages": [
         "Kiss and Tell is a 1945 American comedy film starring Shirley Temple as Corliss Archer.",
         "Shirley Temple Black served as the United States Chief of Protocol from 1976 to 1977.",
     ],
     "titles": ["Kiss and Tell (film)", "Shirley Temple"],
     "gold_titles": ["Kiss and Tell (film)", "Shirley Temple"]},

    {"question": "What nationality was the director of the film Godzilla vs. Biollante?",
     "answer": "Japanese",
     "passages": [
         "Godzilla vs. Biollante is a 1989 Japanese science fiction film directed by Kazuki Ohmori.",
         "Kazuki Ohmori is a Japanese film director.",
     ],
     "titles": ["Godzilla vs. Biollante", "Kazuki Ohmori"],
     "gold_titles": ["Godzilla vs. Biollante", "Kazuki Ohmori"]},

    {"question": "Which film has the director born first, Two Weeks in Another Town or The Spectre of Edgar Allan Poe?",
     "answer": "Two Weeks in Another Town",
     "passages": [
         "Two Weeks in Another Town is a 1962 film directed by Vincente Minnelli, born in 1903.",
         "The Spectre of Edgar Allan Poe is a 1974 film directed by Mohy Quandour, born in 1930.",
     ],
     "titles": ["Two Weeks in Another Town", "The Spectre of Edgar Allan Poe"],
     "gold_titles": ["Two Weeks in Another Town", "The Spectre of Edgar Allan Poe"]},

    {"question": "What is the name of the artist who released the album 'Whenever You Need Somebody' and was found guilty of tax evasion?",
     "answer": "Rick Astley",
     "passages": [
         "'Whenever You Need Somebody' is the debut studio album by English singer Rick Astley, released in 1987.",
         "Rick Astley was found guilty of tax evasion in a UK court.",
     ],
     "titles": ["Whenever You Need Somebody", "Rick Astley"],
     "gold_titles": ["Whenever You Need Somebody", "Rick Astley"]},

    {"question": "In what country did the religion that teaches the Four Noble Truths originate?",
     "answer": "India",
     "passages": [
         "The Four Noble Truths are a central teaching of Buddhism.",
         "Buddhism originated in ancient India, in the eastern Gangetic plain.",
     ],
     "titles": ["Four Noble Truths", "Buddhism"],
     "gold_titles": ["Four Noble Truths", "Buddhism"]},

    {"question": "What is the capital of the country where the Great Barrier Reef is located?",
     "answer": "Canberra",
     "passages": [
         "The Great Barrier Reef is located in the Coral Sea, off the coast of Queensland, Australia.",
         "Canberra is the capital city of Australia.",
     ],
     "titles": ["Great Barrier Reef", "Australia"],
     "gold_titles": ["Great Barrier Reef", "Australia"]},

    {"question": "Which mountain range contains the peak that was first climbed by Edmund Hillary and Tenzing Norgay?",
     "answer": "Himalayas",
     "passages": [
         "Mount Everest was first climbed on 29 May 1953 by Edmund Hillary and Tenzing Norgay.",
         "Mount Everest is part of the Himalayan mountain range.",
     ],
     "titles": ["Mount Everest", "Himalayas"],
     "gold_titles": ["Mount Everest", "Himalayas"]},

    {"question": "Who founded the company that produces the iPhone?",
     "answer": "Steve Jobs",
     "passages": [
         "The iPhone is a line of smartphones designed and marketed by Apple Inc.",
         "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976.",
     ],
     "titles": ["iPhone", "Apple Inc."],
     "gold_titles": ["iPhone", "Apple Inc."]},

    {"question": "What language is primarily spoken in the country where the Eiffel Tower is located?",
     "answer": "French",
     "passages": [
         "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.",
         "French is the official language of France.",
     ],
     "titles": ["Eiffel Tower", "France"],
     "gold_titles": ["Eiffel Tower", "France"]},

    {"question": "What ocean lies to the east of the continent where the Amazon River is found?",
     "answer": "Atlantic Ocean",
     "passages": [
         "The Amazon River flows through South America.",
         "The Atlantic Ocean lies to the east of South America.",
     ],
     "titles": ["Amazon River", "South America"],
     "gold_titles": ["Amazon River", "South America"]},

    {"question": "The director of the 1994 film Pulp Fiction also directed which earlier crime film set in Los Angeles?",
     "answer": "Reservoir Dogs",
     "passages": [
         "Pulp Fiction is a 1994 crime film written and directed by Quentin Tarantino.",
         "Reservoir Dogs is a 1992 crime film written and directed by Quentin Tarantino, set in Los Angeles.",
     ],
     "titles": ["Pulp Fiction", "Reservoir Dogs"],
     "gold_titles": ["Pulp Fiction", "Reservoir Dogs"]},

    {"question": "In what city was the person born who invented the telephone?",
     "answer": "Edinburgh",
     "passages": [
         "Alexander Graham Bell is credited with inventing the first practical telephone.",
         "Alexander Graham Bell was born in Edinburgh, Scotland, on March 3, 1847.",
     ],
     "titles": ["Telephone", "Alexander Graham Bell"],
     "gold_titles": ["Telephone", "Alexander Graham Bell"]},

    {"question": "What was the nationality of the physicist who developed the theory of general relativity?",
     "answer": "German",
     "passages": [
         "The theory of general relativity was developed by Albert Einstein.",
         "Albert Einstein was a German-born theoretical physicist.",
     ],
     "titles": ["General relativity", "Albert Einstein"],
     "gold_titles": ["General relativity", "Albert Einstein"]},

    {"question": "Which university did the founder of Microsoft attend before dropping out?",
     "answer": "Harvard University",
     "passages": [
         "Microsoft was founded by Bill Gates and Paul Allen.",
         "Bill Gates enrolled at Harvard University in 1973 but dropped out in 1975.",
     ],
     "titles": ["Microsoft", "Bill Gates"],
     "gold_titles": ["Microsoft", "Bill Gates"]},

    {"question": "What is the tallest building in the country whose capital is Tokyo?",
     "answer": "Abeno Harukas",
     "passages": [
         "Tokyo is the capital city of Japan.",
         "Abeno Harukas, standing at 300 metres, is the tallest building in Japan.",
     ],
     "titles": ["Tokyo", "Abeno Harukas"],
     "gold_titles": ["Tokyo", "Abeno Harukas"]},

    {"question": "What sport did the athlete play who set the 100m world record that stood from 1999 to 2008?",
     "answer": "athletics",
     "passages": [
         "The 100m world record of 9.79 seconds set in 1999 was held by Maurice Greene until 2008.",
         "Maurice Greene is an American sprinter who competed in track and field athletics.",
     ],
     "titles": ["100 metres world record progression", "Maurice Greene (sprinter)"],
     "gold_titles": ["100 metres world record progression", "Maurice Greene (sprinter)"]},

    {"question": "Who wrote the novel that the film The Shining is based on?",
     "answer": "Stephen King",
     "passages": [
         "The Shining is a 1980 horror film directed by Stanley Kubrick, based on a novel.",
         "The Shining novel was written by Stephen King and published in 1977.",
     ],
     "titles": ["The Shining (film)", "The Shining (novel)"],
     "gold_titles": ["The Shining (film)", "The Shining (novel)"]},

    {"question": "What is the currency of the country that has the largest population in South America?",
     "answer": "Brazilian real",
     "passages": [
         "Brazil is the most populous country in South America, with over 215 million people.",
         "The official currency of Brazil is the Brazilian real.",
     ],
     "titles": ["Brazil", "Brazilian real"],
     "gold_titles": ["Brazil", "Brazilian real"]},

    {"question": "In what year was the company founded that created the search engine used to find 'the world's information'?",
     "answer": "1998",
     "passages": [
         "Google's mission is to organise the world's information and make it universally accessible.",
         "Google was founded on September 4, 1998, by Larry Page and Sergey Brin.",
     ],
     "titles": ["Google Search", "Google"],
     "gold_titles": ["Google Search", "Google"]},

    {"question": "What is the home country of the tennis player who won the most Grand Slam singles titles in the Open Era before Novak Djokovic?",
     "answer": "Switzerland",
     "passages": [
         "Before Novak Djokovic surpassed him, Roger Federer held the record for most Grand Slam singles titles in the Open Era.",
         "Roger Federer is a Swiss professional tennis player.",
     ],
     "titles": ["Grand Slam (tennis)", "Roger Federer"],
     "gold_titles": ["Grand Slam (tennis)", "Roger Federer"]},

    {"question": "What river runs through the city where the Colosseum is located?",
     "answer": "Tiber",
     "passages": [
         "The Colosseum is an oval amphitheatre in the centre of Rome, Italy.",
         "The Tiber is the third-longest river in Italy, running through Rome.",
     ],
     "titles": ["Colosseum", "Rome"],
     "gold_titles": ["Colosseum", "Rome"]},

    {"question": "What element is the main component of the Sun?",
     "answer": "hydrogen",
     "passages": [
         "The Sun is the star at the center of the Solar System.",
         "Hydrogen is by far the most abundant element in the Sun, accounting for about 73% of its mass.",
     ],
     "titles": ["Sun", "Stellar composition"],
     "gold_titles": ["Sun", "Stellar composition"]},

    {"question": "The painter of the Mona Lisa was born in which Italian town?",
     "answer": "Vinci",
     "passages": [
         "The Mona Lisa was painted by Leonardo da Vinci.",
         "Leonardo da Vinci was born on 15 April 1452 in Anchiano, near the town of Vinci, Italy.",
     ],
     "titles": ["Mona Lisa", "Leonardo da Vinci"],
     "gold_titles": ["Mona Lisa", "Leonardo da Vinci"]},

    {"question": "What programming language was created by Guido van Rossum and first released in 1991?",
     "answer": "Python",
     "passages": [
         "Python was created by Guido van Rossum and first released in 1991.",
         "Python is a high-level, general-purpose programming language emphasising readability.",
     ],
     "titles": ["Python (programming language)", "Guido van Rossum"],
     "gold_titles": ["Python (programming language)", "Guido van Rossum"]},
]

_MUSIQUE_QUESTIONS: list[dict] = [
    {"question": "Who is the spouse of the president whose administration introduced the New Deal?",
     "answer": "Eleanor Roosevelt",
     "passages": [
         "The New Deal was a series of programs introduced by President Franklin D. Roosevelt between 1933 and 1939.",
         "Eleanor Roosevelt was the wife of Franklin D. Roosevelt and served as First Lady from 1933 to 1945.",
     ],
     "titles": ["New Deal", "Eleanor Roosevelt"],
     "gold_titles": ["New Deal", "Eleanor Roosevelt"]},

    {"question": "In which country is the birthplace of the inventor of the World Wide Web?",
     "answer": "England",
     "passages": [
         "The World Wide Web was invented by Tim Berners-Lee.",
         "Tim Berners-Lee was born in London, England, on 8 June 1955.",
     ],
     "titles": ["World Wide Web", "Tim Berners-Lee"],
     "gold_titles": ["World Wide Web", "Tim Berners-Lee"]},

    {"question": "What is the name of the university located in the city where the Declaration of Independence was signed?",
     "answer": "University of Pennsylvania",
     "passages": [
         "The Declaration of Independence was signed in Philadelphia, Pennsylvania, in 1776.",
         "The University of Pennsylvania is a private Ivy League university located in Philadelphia.",
     ],
     "titles": ["Declaration of Independence", "University of Pennsylvania"],
     "gold_titles": ["Declaration of Independence", "University of Pennsylvania"]},

    {"question": "What is the nationality of the composer of the opera that features the aria 'Nessun Dorma'?",
     "answer": "Italian",
     "passages": [
         "'Nessun Dorma' is an aria from the opera Turandot.",
         "Turandot was composed by Giacomo Puccini, an Italian opera composer.",
     ],
     "titles": ["Nessun dorma", "Giacomo Puccini"],
     "gold_titles": ["Nessun dorma", "Giacomo Puccini"]},

    {"question": "What is the capital of the country where the company that makes PlayStation was founded?",
     "answer": "Tokyo",
     "passages": [
         "PlayStation is a video gaming brand created and owned by Sony Interactive Entertainment.",
         "Sony was founded in Tokyo, Japan, which is also the country's capital.",
     ],
     "titles": ["PlayStation", "Sony"],
     "gold_titles": ["PlayStation", "Sony"]},

    {"question": "What ocean did the first person to walk on the Moon cross to reach the launch site?",
     "answer": "Atlantic Ocean",
     "passages": [
         "Neil Armstrong was the first person to walk on the Moon, during the Apollo 11 mission in 1969.",
         "Apollo 11 launched from Kennedy Space Center in Florida, USA, which borders the Atlantic Ocean.",
     ],
     "titles": ["Neil Armstrong", "Apollo 11"],
     "gold_titles": ["Neil Armstrong", "Apollo 11"]},

    {"question": "What language is spoken in the country where the highest-grossing film of all time was produced?",
     "answer": "English",
     "passages": [
         "Avatar (2009), directed by James Cameron, is often cited as the highest-grossing film of all time.",
         "Avatar was produced in the United States, where English is the primary language.",
     ],
     "titles": ["Avatar (2009 film)", "James Cameron"],
     "gold_titles": ["Avatar (2009 film)", "James Cameron"]},

    {"question": "What is the home country of the author of the Harry Potter series?",
     "answer": "United Kingdom",
     "passages": [
         "The Harry Potter series was written by J. K. Rowling.",
         "J. K. Rowling is a British author born in Yate, Gloucestershire, England.",
     ],
     "titles": ["Harry Potter", "J. K. Rowling"],
     "gold_titles": ["Harry Potter", "J. K. Rowling"]},

    {"question": "Who was the Prime Minister of the UK when the channel tunnel opened?",
     "answer": "John Major",
     "passages": [
         "The Channel Tunnel officially opened on 6 May 1994.",
         "John Major served as Prime Minister of the United Kingdom from 1990 to 1997.",
     ],
     "titles": ["Channel Tunnel", "John Major"],
     "gold_titles": ["Channel Tunnel", "John Major"]},

    {"question": "What sport is played in the stadium named after the city that hosted the 1992 Summer Olympics?",
     "answer": "football",
     "passages": [
         "The 1992 Summer Olympics were hosted in Barcelona, Spain.",
         "Camp Nou, located in Barcelona, is a football stadium and home of FC Barcelona.",
     ],
     "titles": ["1992 Summer Olympics", "Camp Nou"],
     "gold_titles": ["1992 Summer Olympics", "Camp Nou"]},

    {"question": "What is the birth country of the CEO of Tesla as of 2024?",
     "answer": "South Africa",
     "passages": [
         "Elon Musk has been CEO of Tesla since 2008.",
         "Elon Musk was born on June 28, 1971, in Pretoria, South Africa.",
     ],
     "titles": ["Tesla Inc.", "Elon Musk"],
     "gold_titles": ["Tesla Inc.", "Elon Musk"]},

    {"question": "What is the currency used in the country where Shakespeare was born?",
     "answer": "pound sterling",
     "passages": [
         "William Shakespeare was born in Stratford-upon-Avon, England, in 1564.",
         "The pound sterling is the official currency of the United Kingdom, including England.",
     ],
     "titles": ["William Shakespeare", "Pound sterling"],
     "gold_titles": ["William Shakespeare", "Pound sterling"]},

    {"question": "Who invented the device that Alexander Fleming used to observe penicillin?",
     "answer": "Antonie van Leeuwenhoek",
     "passages": [
         "Alexander Fleming discovered penicillin in 1928 by observing mould under a microscope.",
         "The microscope was invented by Antonie van Leeuwenhoek in the 17th century.",
     ],
     "titles": ["Penicillin", "Microscope"],
     "gold_titles": ["Penicillin", "Microscope"]},

    {"question": "What is the official language of the country that won the FIFA World Cup in 2018?",
     "answer": "French",
     "passages": [
         "France won the 2018 FIFA World Cup, held in Russia.",
         "The official language of France is French.",
     ],
     "titles": ["2018 FIFA World Cup", "France"],
     "gold_titles": ["2018 FIFA World Cup", "France"]},

    {"question": "What continent is the country of the person who first described the theory of evolution by natural selection?",
     "answer": "Europe",
     "passages": [
         "The theory of evolution by natural selection was first described by Charles Darwin.",
         "Charles Darwin was born in Shrewsbury, England, in 1809. England is in Europe.",
     ],
     "titles": ["Natural selection", "Charles Darwin"],
     "gold_titles": ["Natural selection", "Charles Darwin"]},

    {"question": "What is the name of the river that flows through the city where the 2016 Summer Olympics were held?",
     "answer": "Guanabara Bay",
     "passages": [
         "The 2016 Summer Olympics were held in Rio de Janeiro, Brazil.",
         "Rio de Janeiro is located on Guanabara Bay, and is crossed by several rivers including the Carioca River.",
     ],
     "titles": ["2016 Summer Olympics", "Rio de Janeiro"],
     "gold_titles": ["2016 Summer Olympics", "Rio de Janeiro"]},

    {"question": "What is the name of the space agency that employs the astronaut who first orbited the Earth?",
     "answer": "NASA",
     "passages": [
         "John Glenn became the first American to orbit Earth on February 20, 1962.",
         "John Glenn was a NASA astronaut and later a United States Senator.",
     ],
     "titles": ["John Glenn", "NASA"],
     "gold_titles": ["John Glenn", "NASA"]},

    {"question": "What instrument did the musician play who wrote 'Bohemian Rhapsody'?",
     "answer": "piano",
     "passages": [
         "'Bohemian Rhapsody' was written by Freddie Mercury, the lead vocalist of Queen.",
         "Freddie Mercury was an accomplished pianist and primarily played piano as his instrument.",
     ],
     "titles": ["Bohemian Rhapsody", "Freddie Mercury"],
     "gold_titles": ["Bohemian Rhapsody", "Freddie Mercury"]},

    {"question": "Which continent was the explorer born on who first circumnavigated the globe?",
     "answer": "Europe",
     "passages": [
         "Ferdinand Magellan led the first expedition to circumnavigate the Earth.",
         "Ferdinand Magellan was born in Sabrosa, Portugal, in 1480. Portugal is in Europe.",
     ],
     "titles": ["Ferdinand Magellan", "Circumnavigation"],
     "gold_titles": ["Ferdinand Magellan", "Circumnavigation"]},

    {"question": "What is the name of the political party of the first female Prime Minister of the UK?",
     "answer": "Conservative Party",
     "passages": [
         "Margaret Thatcher was the first female Prime Minister of the United Kingdom.",
         "Margaret Thatcher was a member of the Conservative Party.",
     ],
     "titles": ["Margaret Thatcher", "Conservative Party (UK)"],
     "gold_titles": ["Margaret Thatcher", "Conservative Party (UK)"]},

    {"question": "What is the name of the sea that borders the country where the Great Pyramid of Giza is located?",
     "answer": "Red Sea",
     "passages": [
         "The Great Pyramid of Giza is located in Egypt.",
         "Egypt is bordered to the east by the Red Sea.",
     ],
     "titles": ["Great Pyramid of Giza", "Egypt"],
     "gold_titles": ["Great Pyramid of Giza", "Egypt"]},

    {"question": "What was the profession of the father of the person who painted the Sistine Chapel ceiling?",
     "answer": "stonecutter",
     "passages": [
         "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512.",
         "Michelangelo's father, Lodovico di Leonardo Buonarroti Simoni, worked as a stonecutter and small-scale banker.",
     ],
     "titles": ["Sistine Chapel ceiling", "Michelangelo"],
     "gold_titles": ["Sistine Chapel ceiling", "Michelangelo"]},

    {"question": "What sport did the country win a gold medal in that first hosted the modern Olympic Games?",
     "answer": "athletics",
     "passages": [
         "The first modern Olympic Games were hosted by Greece in Athens in 1896.",
         "Greece won several gold medals in athletics at the 1896 Athens Olympics.",
     ],
     "titles": ["1896 Summer Olympics", "Greece at the Olympics"],
     "gold_titles": ["1896 Summer Olympics", "Greece at the Olympics"]},

    {"question": "What is the name of the law of physics formulated by the scientist after whom the SI unit of force is named?",
     "answer": "Newton's laws of motion",
     "passages": [
         "The SI unit of force is the newton, named after Sir Isaac Newton.",
         "Isaac Newton formulated Newton's laws of motion, which describe the relationship between force and motion.",
     ],
     "titles": ["Newton (unit)", "Isaac Newton"],
     "gold_titles": ["Newton (unit)", "Isaac Newton"]},

    {"question": "What is the name of the airline headquartered in the city that is the capital of the UAE?",
     "answer": "Etihad Airways",
     "passages": [
         "Abu Dhabi is the capital city of the United Arab Emirates (UAE).",
         "Etihad Airways is the national airline of the UAE, headquartered in Abu Dhabi.",
     ],
     "titles": ["Abu Dhabi", "Etihad Airways"],
     "gold_titles": ["Abu Dhabi", "Etihad Airways"]},
]


def _auto_generate_question_files() -> None:
    """
    Write hotpotqa_sample.json and musique_sample.json into testing_questions/
    if the folder is empty (or doesn't exist yet).  Called automatically
    before _discover_files() so the benchmark can always run out-of-the-box.
    """
    QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)

    has_files = any(
        p.suffix.lower() in {".json", ".txt"}
        for p in QUESTIONS_DIR.iterdir()
        if p.is_file()
    )
    if has_files:
        return  # folder already populated — nothing to do

    print(
        f"  [AUTO] testing_questions/ is empty.\n"
        f"         Generating sample benchmark files …"
    )

    for fname, data in [
        ("hotpotqa_sample.json", _HOTPOTQA_QUESTIONS),
        ("musique_sample.json",  _MUSIQUE_QUESTIONS),
    ]:
        out = QUESTIONS_DIR / fname
        with out.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(f"         ✓  {fname}  ({len(data)} questions)")

    print()


def _discover_files() -> list[Path]:
    """Return all .json and .txt files in testing_questions/.
    Auto-generates sample files first if the folder is empty."""
    _auto_generate_question_files()   # ← no-op when files already exist

    files = sorted(
        p for p in QUESTIONS_DIR.iterdir()
        if p.suffix.lower() in {".json", ".txt"} and p.is_file()
    )
    if not files:
        raise FileNotFoundError(
            f"\n[ERROR] No .json or .txt files found in {QUESTIONS_DIR}.\n"
            "  This should not happen — please check folder permissions."
        )
    return files


def _load_json_file(path: Path, n: int) -> list[dict]:
    """
    Load a JSON file containing a list of question objects.

    Required field  : question  (str)
    Optional fields : answer (str), passages (list[str]),
                      titles (list[str]), gold_titles (list[str])
    """
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: top-level JSON must be an array.")

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
    return items


def _load_txt_file(path: Path, n: int) -> list[dict]:
    """
    Load a plain-text file of questions.

    Accepted line formats (one entry per line):
      • question only            →  answer = ""
      • question<TAB>answer      →  tab-separated
      • question | answer        →  pipe-separated
    Blank lines and lines starting with '#' are ignored.
    """
    items = []
    with path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            if len(items) >= n:
                break
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if "\t" in line:
                parts = line.split("\t", 1)
                question, answer = parts[0].strip(), parts[1].strip()
            elif " | " in line:
                parts = line.split(" | ", 1)
                question, answer = parts[0].strip(), parts[1].strip()
            else:
                question, answer = line, ""

            items.append({
                "question":    question,
                "answer":      answer,
                "passages":    [],
                "titles":      [],
                "gold_titles": [],
            })
    return items


def _load_file(path: Path, n: int) -> list[dict]:
    suffix = path.suffix.lower()
    print(f"  Loading  {path.name}  …")
    if suffix == ".json":
        items = _load_json_file(path, n)
    elif suffix == ".txt":
        items = _load_txt_file(path, n)
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


# Strip <think>…</think> blocks emitted by Qwen3 and DeepSeek thinking mode
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> reasoning preambles."""
    return _THINK_RE.sub("", text).strip()


# ── Adaptive (Ours) orchestrator caller ───────────────────────────────────────

def _build_adaptive_caller() -> Callable[[str], tuple[str, float]]:
    from llms.custom.orchestrator import (
        classify_task,
        select_primary_model,
        build_specialist_prompt,
        build_synthesis_prompt,
        should_synthesize,
        FALLBACK_MODEL,
    )
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
        "qwen2.5-coder-7b":   "qwen2.5-coder:7b",
        "qwen3.5-0.8b":       "qwen3.5:0.8b",
    }

    _llm_cache: dict[str, ChatOllama] = {}

    def _get_llm(model_key: str) -> ChatOllama:
        if model_key not in _llm_cache:
            tag = OLLAMA_TAG_MAP.get(model_key)
            if not tag:
                print(f"  [WARN] Orchestrator: unknown key '{model_key}', "
                      f"falling back to falcon3.")
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
            print(f"  [ERROR] Orchestrator primary '{primary_key}': {exc}")
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
            except Exception as exc:
                print(f"  [WARN] Synthesis failed: {exc}")

        lat = (time.perf_counter() - t0) * 1000.0
        return raw, lat

    return call


# ══════════════════════════════════════════════════════════════════════════════
#  QA prompt
# ══════════════════════════════════════════════════════════════════════════════

def _qa_prompt(question: str, passages: list[str]) -> str:
    if passages:
        context = "\n\n".join(
            f"[{i + 1}] {p}" for i, p in enumerate(passages[:4])
        )
        context_block = f"Passages:\n{context}\n\n"
    else:
        # No retrieved passages — ask the model from its own knowledge
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
        return 0.0
    gold_set = {t.lower() for t in gold}
    return sum(1 for t in retrieved[:k] if t.lower() in gold_set) / k


# ── Simulated retrieval ───────────────────────────────────────────────────────

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
    scores = [
        len(q_tokens & set(_normalise(f"{t} {p}").split()))
        for t, p in zip(titles, passages)
    ]
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return (
        [titles[i] for i in ranked[:k]],
        [passages[i] for i in ranked[:k]],
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
        if (idx + 1) % 100 == 0:
            running_em = 100 * sum(em_list) / max(len(em_list), 1)
            print(f"     … {idx + 1}/{n}  (running EM: {running_em:.1f}%)")

        question = item["question"]
        gold_answer = item["answer"]
        titles = item["titles"]
        passages = item["passages"]
        gold_titles = item["gold_titles"]

        # Simulated retrieval — rank passages, keep top-4
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

        # 1. Strip <think>…</think> blocks (Qwen3 / DeepSeek)
        pred = _strip_thinking(pred)

        # 2. Take only the first line (ignore multi-line reasoning spillover)
        pred = pred.split("\n")[0].strip()

        # 3. Strip common preamble phrases
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
        "em":  round(100 * sum(em_list) / max(len(em_list),  1), 1),
        "f1":  round(100 * sum(f1_list) / max(len(f1_list),  1), 1),
        "p4":  round(sum(p4_list) / max(len(p4_list),  1), 3),
        "lat": round(sum(lat_list) / max(len(lat_list), 1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Result table
# ══════════════════════════════════════════════════════════════════════════════

_COL = 26


def _print_table(
    dataset_name: str,
    results: list[tuple[str, dict]],
    n_questions: int,
) -> None:
    print()
    print("═" * 72)
    print(f"  {dataset_name}   ({n_questions} questions)")
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


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║     LLM Benchmark — local testing_questions/ folder             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Max questions per file : {N_QUESTIONS}")
    print(f"  Questions directory    : {QUESTIONS_DIR}")
    print()

    # ── Discover question files ────────────────────────────────────────────────
    all_files = _discover_files()

    # Apply --datasets filter if given
    if args.datasets:
        filter_lower = {d.lower() for d in args.datasets}
        selected_files = [
            p for p in all_files
            if p.stem.lower() in filter_lower
        ]
        if not selected_files:
            print(
                f"[ERROR] None of the requested datasets "
                f"({', '.join(args.datasets)}) matched files in "
                f"{QUESTIONS_DIR}.\n"
                f"  Available: {', '.join(p.stem for p in all_files)}"
            )
            sys.exit(1)
    else:
        selected_files = all_files

    print(
        f"  Dataset files selected : {', '.join(p.name for p in selected_files)}")
    print()

    # ── Ollama check ───────────────────────────────────────────────────────────
    needs_ollama = any(
        ollama_id not in ("adaptive",)
        for _, ollama_id, _ in MODELS
        if not any(s.lower() in _.lower() for s in args.skip)
        if not args.only or any(o.lower() in _.lower() for o in args.only)
    )
    if needs_ollama:
        _check_ollama()

    # ── Initialise model callers ───────────────────────────────────────────────
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
            print(f"           Skipping {display_name}.")

    if not active:
        print("\n[ERROR] No models could be initialised. Exiting.")
        sys.exit(1)

    print(f"\n  {len(active)} model(s) ready.\n")

    # ── Dataset / file loop ────────────────────────────────────────────────────
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

    # ── Final tables ───────────────────────────────────────────────────────────
    print("\n\n" + "═" * 72)
    print("  FINAL RESULTS")
    print("═" * 72)
    for ds_name, results in all_results.items():
        _print_table(ds_name, results, N_QUESTIONS)

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
