TASK=${1:-cfa}
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_ENV="coact311"
MODEL="Qwen/Qwen2.5-72B-Instruct"
MODEL_ALIAS="Qwen2.5-72B-Instruct"
PORT=8001
TP=4
GPUS="0,1,2,3"
MAX_LEN=5200
MEM_UTIL=0.95
DTYPE="bfloat16"


KW_SLEEP_CALLS=0.2       # sleep between API calls (s)
KW_SLEEP_ITERS=0.5       # sleep between expansion iterations (s)
EXP_ITERS=2              # number of retrieval-extraction rounds
EXP_SAMPLE=3             # keywords sampled per round
EXP_TOPK=2               # top-k docs retrieved per round
EXP_MAXDOCS=3            # max docs sent to LLM per round
QG_PAIRS=5               # max keyword pairs for paired prompts
QG_MAX=20                # total instructions
CF_K=3                   # responses generated per question
CF_THRESHOLD=0.5         # min consistency to keep a question

SKIP_KW=false; SKIP_EXP=false; SERVER_ONLY=false; NO_SERVER=false
shift 2>/dev/null || true
for arg in "$@"; do
    case $arg in
        --skip-keywords)  SKIP_KW=true     ;;
        --skip-expansion) SKIP_EXP=true    ;;
        --server-only)    SERVER_ONLY=true ;;
        --no-server)      NO_SERVER=true   ;;
    esac
done

unset USE_AZURE_OPENAI 2>/dev/null || true
export VLLM_API_BASE="http://localhost:${PORT}/v1"
API_URL="${VLLM_API_BASE}/models"

# ── vLLM server ──────────────────────────────────────────────────
if [[ "$NO_SERVER" == false ]]; then
    if ! curl -sf "${API_URL}" >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES=${GPUS} VLLM_WORKER_MULTIPROC_METHOD=spawn \
            conda run --no-capture-output -n "${CONDA_ENV}" \
            python -m vllm.entrypoints.openai.api_server \
                --model "${MODEL}" --served-model-name "${MODEL_ALIAS}" \
                --host 0.0.0.0 --port "${PORT}" \
                --tensor-parallel-size "${TP}" --max-model-len "${MAX_LEN}" \
                --gpu-memory-utilization "${MEM_UTIL}" --dtype "${DTYPE}" \
                --trust-remote-code --enable-prefix-caching &
        echo "$!" > "${PROJ}/.vllm.pid"
        elapsed=0
        while ! curl -sf "${API_URL}" >/dev/null 2>&1; do
            sleep 5; elapsed=$((elapsed+5))
            [[ $elapsed -ge 600 ]] && exit 1
        done
    fi
fi
[[ "$SERVER_ONLY" == true ]] && exit 0

PY="conda run -n ${CONDA_ENV} python"

[[ "$SKIP_KW" == false ]] && \
    $PY "${PROJ}/scripts/keywords_generator.py" --task "${TASK}" \
        --sleep_between_calls "${KW_SLEEP_CALLS}" --sleep_between_iterations "${KW_SLEEP_ITERS}"

[[ "$SKIP_EXP" == false ]] && \
    $PY "${PROJ}/scripts/keyword_expansion_from_corpus.py" --task "${TASK}" \
        --num_iterations "${EXP_ITERS}" --num_sample_keywords "${EXP_SAMPLE}" \
        --top_k "${EXP_TOPK}" --max_docs "${EXP_MAXDOCS}"

$PY "${PROJ}/scripts/generate_questions.py" --task "${TASK}" \
    --max_pairs "${QG_PAIRS}" --max_instructions "${QG_MAX}"

$PY "${PROJ}/scripts/consistency_filter.py" --task "${TASK}" \
    --k "${CF_K}" --threshold "${CF_THRESHOLD}"
