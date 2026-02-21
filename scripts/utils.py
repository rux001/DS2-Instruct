"""Shared utilities: answer extraction, consistency scoring, keyword parsing."""

import re
from collections import Counter

MATH_TASKS  = {"math"}
GSM8K_TASKS = {"gsm8k"}
MC_TASKS    = {"cfa", "logiqa", "gpqa", "medqa"}
QA_TASKS    = {"pubmedqa"}


def extract_math_answer(resp):
    resp = resp.replace(",", "")
    m = re.findall(r"\\boxed\{(.*?)\}", resp)
    if m:
        return m[-1].strip()
    m = re.findall(r"final answer is:?\s*(.*)", resp, re.IGNORECASE)
    if m:
        return m[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", resp)
    return nums[-1] if nums else None


def extract_gsm8k_answer(resp):
    m = re.findall(r"[Ff]inal [Aa]nswer:?\s*(.+)", resp)
    if m:
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", m[-1])
        return nums[0].replace(",", "") if nums else m[-1].strip()
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", resp)
    return nums[-1].replace(",", "") if nums else None


def extract_mc_answer(resp):
    m = re.search(r"[Aa]nswer:\s*([A-D])", resp)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", resp)
    return m.group(1) if m else None


def extract_qa_answer(resp):
    m = re.search(r"[Aa]nswer:\s*(yes|no|maybe)", resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    low = resp.strip().lower()
    for ans in ("yes", "no", "maybe"):
        if low.startswith(ans):
            return ans
    return None


def get_extractor(task):
    if task in MATH_TASKS:  return extract_math_answer
    if task in GSM8K_TASKS: return extract_gsm8k_answer
    if task in MC_TASKS:    return extract_mc_answer
    if task in QA_TASKS:    return extract_qa_answer
    return None


def consistency(answers):
    valid = [a for a in answers if a is not None]
    if not valid:
        return 0.0
    return Counter(valid).most_common(1)[0][1] / len(answers)


def parse_keywords(response):
    if not response:
        return []
    response = re.sub(r'^(keywords?|concepts?|terms?):\s*', '', response, flags=re.IGNORECASE)
    cleaned = []
    for kw in response.split(','):
        kw = kw.strip().lower()
        kw = re.sub(r'[\s-]+', '_', kw)
        kw = re.sub(r'[^a-z0-9_.]', '', kw).strip('_')
        if kw:
            cleaned.append(kw)
    return list(dict.fromkeys(cleaned))
