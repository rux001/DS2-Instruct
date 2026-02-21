import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import itertools
import argparse
from tqdm import tqdm

import config
import prompts
from model_backends import setup_model_backend, call_model


def load_keywords(task):
    path = config.TASK_CONFIGS[task]["expanded_keywords_file"]
    with open(path) as f:
        return json.load(f).get("all_keywords", [])


def build_prompts(task, max_pairs, max_total):
    desc = prompts.TASK_DESCRIPTIONS[task]
    keywords = load_keywords(task)
    print(f"Loaded {len(keywords)} keywords")

    out = []

    for kw in keywords:
        for qt, qt_desc in prompts.QUERY_TYPES.items():
            out.append(prompts.SINGLE_KEYWORD_PROMPT_TEMPLATE.format(
                task_description=desc, keyword=kw,
                query_type=qt, query_type_description=qt_desc))

    pairs = list(itertools.combinations(keywords, 2))
    random.shuffle(pairs)
    for kw1, kw2 in pairs[:max_pairs]:
        for qt, qt_desc in prompts.QUERY_TYPES.items():
            out.append(prompts.PAIRED_KEYWORD_PROMPT_TEMPLATE.format(
                task_description=desc, keyword1=kw1, keyword2=kw2,
                query_type=qt, query_type_description=qt_desc))

    random.shuffle(out)
    return out[:max_total]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=config.TASK_CONFIGS.keys())
    ap.add_argument("--max_pairs", type=int, default=2000)
    ap.add_argument("--max_instructions", type=int, default=8000)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    templates = build_prompts(args.task, args.max_pairs, args.max_instructions)
    print(f"Built {len(templates)} prompt templates")

    setup_model_backend()
    questions = []
    for t in tqdm(templates, desc="Generating questions"):
        resp = call_model(t, args.max_tokens, args.temperature)
        if resp and resp.strip():
            questions.append({"question": resp.strip(), "template": t})

    out = os.path.join(config.OUTPUT_DIR, f"{args.task}_questions.json")
    with open(out, "w") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"\n{len(questions)}/{len(templates)} questions → {out}")


if __name__ == "__main__":
    main()
