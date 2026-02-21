import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
from tqdm import tqdm

import config
import prompts
from model_backends import setup_model_backend, call_model
from bm25_retriever import load_documents_from_dir, load_keywords, build_query, retrieve_documents
from utils import parse_keywords


def extract_keywords_from_docs(docs, max_docs=100, max_doc_len=3000,
                               max_tokens=150, temperature=0.7):
    prompt_tpl = prompts.get_prompt("wikipedia_extraction")
    found = set()
    for doc in tqdm(docs[:max_docs], desc="Extracting keywords"):
        content = doc[:max_doc_len]
        prompt = prompt_tpl.format(content=content, found_keywords=[], found_keywords_str="")
        resp = call_model(prompt, max_tokens, temperature)
        if resp:
            found.update(parse_keywords(resp))
    return found


def expand_keywords(task, num_iterations=10, num_sample=10, top_k=500,
                    max_docs=100, max_doc_len=3000, max_tokens=150,
                    temperature=0.7, seed=42):
    setup_model_backend()
    random.seed(seed)

    tc = config.TASK_CONFIGS[task]
    initial = load_keywords(tc["output_file"])
    print(f"Loaded {len(initial)} initial keywords")

    all_docs = load_documents_from_dir()
    print(f"Loaded {len(all_docs)} documents")

    current = set(initial)
    task_desc = prompts.TASK_DESCRIPTIONS.get(task, "")
    stats = []

    for it in tqdm(range(num_iterations), desc="Expansion iterations"):
        sampled = random.sample(list(current), min(num_sample, len(current)))
        query = build_query(task_desc, sampled, num_keywords=len(sampled))
        rel_docs = retrieve_documents(all_docs, query, top_k=top_k)
        extracted = extract_keywords_from_docs(rel_docs, max_docs, max_doc_len,
                                               max_tokens, temperature)
        new = extracted - current
        current.update(extracted)
        stats.append({"iteration": it + 1, "new": len(new), "total": len(current)})
        tqdm.write(f"  iter {it+1}: +{len(new)} keywords, total={len(current)}")

    print(f"\nFinal: {len(initial)} initial → {len(current)} total")
    return {
        "all_keywords": sorted(current),
        "initial_keywords": initial,
        "newly_expanded_keywords": sorted(current - set(initial)),
        "iteration_stats": stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=config.TASK_CONFIGS.keys())
    ap.add_argument("--num_iterations", type=int, default=10)
    ap.add_argument("--num_sample_keywords", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--max_docs", type=int, default=100)
    ap.add_argument("--max_doc_length", type=int, default=3000)
    ap.add_argument("--max_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--random_seed", type=int, default=42)
    args = ap.parse_args()

    results = expand_keywords(
        args.task, args.num_iterations, args.num_sample_keywords,
        args.top_k, args.max_docs, args.max_doc_length,
        args.max_tokens, args.temperature, args.random_seed,
    )
    out = config.TASK_CONFIGS[args.task]["expanded_keywords_file"]
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
