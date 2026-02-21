# DS²-INSTRUCT

**Domain-Specific Data Synthesis for Large Language Models Instruction Tuning**


> A zero-shot framework that generates domain-specific instruction-tuning datasets without human supervision. 
## Setup

```bash
conda env create -f environment.yml
conda activate ds2
```


## Usage

### Local vLLM (Qwen 72B)

Run the full pipeline:
```bash
bash shell/run_pipeline_qwen72b.sh cfa
```

### OpenAI

```bash
export USE_AZURE_OPENAI=1
bash shell/run_pipeline_azure_openai.sh cfa
```


## Supported Tasks

| Task | Domain |
|------|--------|
| `cfa` | Finance |
| `gsm8k` | Grade school math |
| `math` | Competition math |
| `pubmedqa` | Biomedicine |
| `logiqa` | Logical reasoning |
| `gpqa` | Graduate-level science |
| `medqa` | Medicine |

## Output

The final output is an SFT dataset in `output/{task}_sft_dataset.json`
