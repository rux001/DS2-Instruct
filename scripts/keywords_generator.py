import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import random
import argparse

import config
import prompts
from model_backends import setup_model_backend, call_model
from utils import parse_keywords


def generate_initial_keywords(count):
    prompt = prompts.get_initial_keywords_prompt().format(count=count)
    response = call_model(prompt, config.MAX_TOKENS_INITIAL, config.TEMPERATURE)
    kw = parse_keywords(response)
    return kw if kw else _fallback_keywords()


def expand_keywords(current, prompt_fn, max_sample):
    sample = random.sample(current, min(len(current), max_sample))
    prompt = prompt_fn().format(count=config.KEYWORDS_PER_ITERATION,
                                keywords_str=", ".join(sample))
    response = call_model(prompt, config.MAX_TOKENS_EXPAND, config.TEMPERATURE)
    kw = parse_keywords(response)
    return kw if kw else _fallback_keywords()


def _fallback_keywords():
    n = min(config.KEYWORDS_PER_ITERATION, len(config.MOCK_KEYWORDS))
    return random.sample(config.MOCK_KEYWORDS, n)


def expand_iteration(iteration, all_kw, sample_n, max_sample, sleep):
    cur = list(all_kw)
    rand = random.sample(cur, min(sample_n, len(cur)))
    data = {"iteration": iteration, "starting_count": len(all_kw), "added": {}}
    new_all = []

    prereqs = expand_keywords(rand, prompts.get_prerequisite_keywords_prompt, max_sample)
    new_pre = [k for k in prereqs if k not in all_kw]
    data["added"]["prerequisites"] = new_pre
    new_all.extend(new_pre)
    time.sleep(sleep)

    advanced = expand_keywords(rand, prompts.get_advanced_keywords_prompt, max_sample)
    merged = all_kw | set(new_pre)
    new_adv = [k for k in advanced if k not in merged]
    data["added"]["advanced"] = new_adv
    new_all.extend(new_adv)

    uniq = list(dict.fromkeys(new_all))
    data["total_added"] = len(uniq)
    data["ending_count"] = len(all_kw) + len(uniq)
    return uniq, data


def save_results(keywords, history, model):
    output = {
        "metadata": {
            "task_name": config.TASK_NAME,
            "domain": config.DOMAIN,
            "total_unique_keywords": len(keywords),
            "initial_keywords_count": len(history.get("initial_keywords", [])),
            "model_used": model,
            "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "backend": config.get_backend_config()["type"],
        },
        "keywords": keywords,
        "generation_history": history,
    }
    with open(config.OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main(task, sample_n=10, max_sample=15, sleep_calls=1.0, sleep_iters=2.0):
    config.load_task_config(task)
    ok, model = setup_model_backend()
    if not ok:
        return

    initial = generate_initial_keywords(config.INITIAL_KEYWORDS_COUNT)
    if not initial:
        return

    all_kw = set(initial)
    history = {"task_name": config.TASK_NAME, "domain": config.DOMAIN,
               "initial_keywords": initial, "expansion_iterations": []}

    for i in range(1, config.EXPANSION_ITERATIONS + 1):
        new, data = expand_iteration(i, all_kw, sample_n, max_sample, sleep_calls)
        history["expansion_iterations"].append(data)
        all_kw.update(new)
        time.sleep(sleep_iters)

    save_results(sorted(all_kw), history, model)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=config.TASK_CONFIGS.keys())
    ap.add_argument("--sample_keywords_count", type=int, default=10)
    ap.add_argument("--max_sample_size", type=int, default=15)
    ap.add_argument("--sleep_between_calls", type=float, default=1.0)
    ap.add_argument("--sleep_between_iterations", type=float, default=2.0)
    a = ap.parse_args()
    main(a.task, a.sample_keywords_count, a.max_sample_size,
         a.sleep_between_calls, a.sleep_between_iterations)
