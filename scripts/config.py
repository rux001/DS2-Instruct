import os
from enum import Enum

# ── Project Paths ────────────────────────────────────────────────

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(PROJECT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Backend ──────────────────────────────────────────────────────

class BackendType(Enum):
    LLAMAFACTORY = "llamafactory"
    VLLM_API     = "vllm_api"
    VLLM_DIRECT  = "vllm_direct"
    AZURE_OPENAI = "azure_openai"

BACKEND = BackendType.AZURE_OPENAI if os.getenv("USE_AZURE_OPENAI") else BackendType.VLLM_API

# ── Model ────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
MODEL_PATH = "Qwen/Qwen2.5-72B-Instruct"

# ── vLLM Server ──────────────────────────────────────────────────

VLLM_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("VLLM_PORT", 8001)),
    "served_model_name": os.getenv("VLLM_SERVED_MODEL_NAME", "Qwen2.5-72B-Instruct"),
    "max_model_len": 5200,
    "pipeline_parallel_size": 1,
    "tensor_parallel_size": int(os.getenv("VLLM_TP_SIZE", 4)),
    "gpu_memory_utilization": 0.95,
    "trust_remote_code": True,
    "dtype": "bfloat16",
    "enable_prefix_caching": True,
    "max_num_batched_tokens": 2048,
}

# ── API ──────────────────────────────────────────────────────────

API_BASE = os.getenv("VLLM_API_BASE", f"http://localhost:{VLLM_CONFIG['port']}/v1")
API_KEY  = os.getenv("VLLM_API_KEY", "EMPTY")

AZURE_OPENAI_CONFIG = {
    "deployment_name": "",
    "api_key": "",
    "endpoint": "",
    "api_version": "",
}

# ── Generation Parameters ────────────────────────────────────────

INITIAL_KEYWORDS_COUNT    = 3
EXPANSION_ITERATIONS      = 2
KEYWORDS_PER_ITERATION    = 2
MIN_KEYWORDS_PER_EXPANSION = 1

MAX_TOKENS_INITIAL = 2048
MAX_TOKENS_EXPAND  = 2048
TEMPERATURE        = 0.7
STOP_TOKENS        = ["<|im_end|>", "<|endoftext|>"]
TOP_P              = 0.7
REQUEST_TIMEOUT    = 120

# ── Default globals (overwritten by load_task_config) ────────────

TASK_NAME         = "gsm8k"
DOMAIN            = "math"
OUTPUT_FILE       = os.path.join(PROJECT_DIR, "gsm8k_keywords.json")
MOCK_KEYWORDS     = ["algebra", "arithmetic", "fractions", "percentages", "geometry",
                     "probability", "word_problems", "unit_conversion", "ratios"]
EXAMPLE_MULTIWORD = "word_problems, unit_conversion"
EXAMPLE_SINGLE    = "algebra, fractions, ratios"

# ── Task Configurations ──────────────────────────────────────────

def _task(name, domain, mock_kw, ex_multi, ex_single):
    return {
        "task_name": name,
        "domain": domain,
        "output_file":              os.path.join(PROJECT_DIR, f"{name}_keywords.json"),
        "expanded_keywords_file": os.path.join(PROJECT_DIR, f"expanded_{name}_keywords.json"),
        "sft_output_file":          os.path.join(OUTPUT_DIR,  f"{name}_sft_dataset.json"),
        "mock_keywords": mock_kw,
        "example_multiword": ex_multi,
        "example_single": ex_single,
    }

TASK_CONFIGS = {
    "cfa": _task("cfa", "finance",
                 ["portfolio_theory", "risk_management", "asset_valuation", "derivatives",
                  "bonds", "equity_analysis", "financial_ratios", "capm", "wacc"],
                 "portfolio_management, risk_assessment", "bonds, equity, capm"),
    "gsm8k": _task("gsm8k", "math",
                   ["algebra", "arithmetic", "fractions", "percentages", "geometry",
                    "probability", "word_problems", "unit_conversion", "ratios"],
                   "word_problems, unit_conversion", "algebra, fractions, ratios"),
    "pubmedqa": _task("pubmedqa", "biomedicine",
                      ["clinical_trials", "gene_expression", "drug_efficacy", "pathophysiology",
                       "epidemiology", "molecular_biology", "immunology", "diagnostic_methods"],
                      "clinical_trials, drug_efficacy", "genetics, immunology, oncology"),
    "logiqa": _task("logiqa", "logical_reasoning",
                    ["deductive_reasoning", "inductive_reasoning", "syllogism", "logical_fallacies",
                     "premise", "conclusion", "analogical_reasoning", "causal_inference"],
                    "deductive_reasoning, logical_fallacies", "syllogism, premise, inference"),
    "gpqa": _task("gpqa", "physics, chemistry, biology",
                  ["quantum_mechanics", "organic_chemistry", "molecular_biology", "general_relativity",
                   "thermodynamics", "spectroscopy", "particle_physics", "gene_editing"],
                  "quantum_mechanics, organic_chemistry", "relativity, genetics, spectroscopy"),
    "math": _task("math", "math",
                  ["algebra", "geometry", "trigonometry", "calculus", "statistics",
                   "probability", "linear_algebra", "differential_equations"],
                  "linear_algebra, differential_equations", "trigonometry, calculus, statistics"),
    "medqa": _task("medqa", "medicine",
                   ["physiology", "pathology", "pharmacology", "clinical_reasoning"],
                   "clinical_reasoning", "pharmacology, physiology"),
}

# BM25 Config ──────────────────────────────────────

WIKIPEDIA_CONFIG = {
    "document_source_dir": os.path.join(PROJECT_DIR, "wikipedia_data"),
    "document_pattern": "wikipedia_en_batch_*",
    "max_articles_to_process": 10000,
    "max_batches": 10,
    "content_field": "webpage",
    "window_size": 50,
    "min_count": 1,
    "top_cooccurring_articles": 100,
    "batch_size": 100,
    "use_parallel_processing": True,
    "max_workers": 4,
    "max_content_length": 3000,
    "llm_max_tokens": 300,
    "llm_temperature": 0.7,
}

# ── Helpers ──────────────────────────────────────────────────────

def load_task_config(task_key):
    """Load a task's config into module-level globals (used by prompts.py)."""
    if task_key not in TASK_CONFIGS:
        raise ValueError(f"Unknown task '{task_key}'. Available: {list(TASK_CONFIGS.keys())}")
    tc = TASK_CONFIGS[task_key]
    globals().update({
        "TASK_NAME":         tc["task_name"],
        "DOMAIN":            tc["domain"],
        "OUTPUT_FILE":       tc["output_file"],
        "MOCK_KEYWORDS":     tc["mock_keywords"],
        "EXAMPLE_MULTIWORD": tc["example_multiword"],
        "EXAMPLE_SINGLE":    tc["example_single"],
    })

def get_backend_config():
    if BACKEND == BackendType.VLLM_DIRECT:
        return {"type": "vllm_direct", "model_path": MODEL_PATH, "vllm_config": VLLM_CONFIG}
    if BACKEND == BackendType.VLLM_API:
        return {"type": "vllm_api", "api_base": API_BASE, "api_key": API_KEY,
                "model_name": VLLM_CONFIG["served_model_name"]}
    if BACKEND == BackendType.AZURE_OPENAI:
        return {"type": "azure_openai", **AZURE_OPENAI_CONFIG}
    return {"type": "llamafactory", "api_base": API_BASE, "api_key": API_KEY,
            "model_name": MODEL_NAME}
