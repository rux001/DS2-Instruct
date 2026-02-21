"""Model backend: routes LLM calls to vLLM API, vLLM direct, or Azure OpenAI."""

import requests
import config

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


class VLLMDirectBackend:
    def __init__(self, model_path, vllm_config):
        self.model_path = model_path
        self.vllm_config = vllm_config
        self.llm = None

    def setup(self):
        if not VLLM_AVAILABLE:
            return False, ""
        self.llm = LLM(
            model=self.model_path,
            max_model_len=self.vllm_config.get("max_model_len", 4096),
            tensor_parallel_size=self.vllm_config.get("tensor_parallel_size", 4),
            gpu_memory_utilization=self.vllm_config.get("gpu_memory_utilization", 0.8),
            trust_remote_code=self.vllm_config.get("trust_remote_code", True),
            dtype=self.vllm_config.get("dtype", "bfloat16"),
            enforce_eager=self.vllm_config.get("enforce_eager", False),
        )
        return True, self.model_path

    def generate(self, prompt, max_tokens, temperature):
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature,
                                top_p=config.TOP_P, stop=config.STOP_TOKENS)
        out = self.llm.generate([prompt], params)
        text = out[0].outputs[0].text.strip()
        return text or None


class APIBackend:
    def __init__(self, api_base, api_key, model_name):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name

    def setup(self):
        resp = requests.get(f"{self.api_base}/models", timeout=120)
        resp.raise_for_status()
        models = [m["id"] for m in resp.json().get("data", [])]
        if self.model_name in models:
            return True, self.model_name
        if models:
            self.model_name = models[0]
            return True, self.model_name
        return False, ""

    def generate(self, prompt, max_tokens, temperature):
        resp = requests.post(
            f"{self.api_base}/chat/completions",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model_name,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature,
                  "top_p": config.TOP_P, "stop": config.STOP_TOKENS,
                  "stream": False},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None


class AzureOpenAIBackend:
    def __init__(self, deployment_name, api_key, endpoint, api_version):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = endpoint

    def setup(self):
        return True, self.deployment_name

    def generate(self, prompt, max_tokens, temperature):
        resp = requests.post(
            self.endpoint,
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            json={"messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=config.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


# ── Public API ───────────────────────────────────────────────────

_backend = None

def setup_model_backend():
    global _backend
    bc = config.get_backend_config()
    if bc["type"] == "vllm_direct":
        _backend = VLLMDirectBackend(bc["model_path"], bc["vllm_config"])
    elif bc["type"] == "azure_openai":
        _backend = AzureOpenAIBackend(bc["deployment_name"], bc["api_key"],
                                      bc["endpoint"], bc.get("api_version", ""))
    else:
        _backend = APIBackend(bc["api_base"], bc["api_key"], bc["model_name"])
    return _backend.setup()

def call_model(prompt, max_tokens, temperature):
    return _backend.generate(prompt, max_tokens, temperature)
