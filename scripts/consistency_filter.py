import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import argparse
from collections import Counter
from tqdm import tqdm

import config
import prompts
from model_backends import setup_model_backend, call_model
from utils import get_extractor, consistency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=config.TASK_CONFIGS.keys())
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperatures", default="0.7,0.8,0.9")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--target_size", type=int, default=6000)
    args = ap.parse_args()

    input_file  = os.path.join(config.OUTPUT_DIR, f"{args.task}_questions.json")
    output_file = config.TASK_CONFIGS[args.task]["sft_output_file"]
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    temps     = [float(t) for t in args.temperatures.split(",")]
    suffix    = prompts.PROMPT_SUFFIXES[args.task]
    extractor = get_extractor(args.task)

    with open(input_file) as f:
        questions = json.load(f)

    setup_model_backend()

    sft, kept, filtered, no_ans = [], 0, 0, 0

    for entry in tqdm(questions, desc="Responses & filtering"):
        question = entry["question"]
        prompt = f"{question}\n\n{suffix}"

        responses = [r for i in range(args.k)
                     if (r := call_model(prompt, args.max_tokens, temps[i % len(temps)]))]

        if not responses:
            no_ans += 1
            continue

        if extractor:
            answers = [extractor(r) for r in responses]
            score = consistency(answers)
            if score < args.threshold:
                filtered += 1
                continue
            valid = [a for a in answers if a is not None]
            majority = Counter(valid).most_common(1)[0][0] if valid else None
            best = next((r for r, a in zip(responses, answers) if a == majority), responses[0])
        else:
            best = responses[0]

        sft.append({"conversations": [
            {"from": "human", "value": question},
            {"from": "gpt",   "value": best},
        ]})
        kept += 1

    if len(sft) > args.target_size:
        random.shuffle(sft)
        sft = sft[:args.target_size]

    with open(output_file, "w") as f:
        json.dump(sft, f, indent=2, ensure_ascii=False)

    print(f"total={len(questions)}  kept={kept}  filtered={filtered}  "
          f"no_answer={no_ans}  sft_size={len(sft)} → {output_file}")


if __name__ == "__main__":
    main()
