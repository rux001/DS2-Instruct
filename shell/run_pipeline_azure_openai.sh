#!/bin/bash
# Run DataSynthesis pipeline using Azure OpenAI backend.
#
# Usage: ./shell/run_pipeline_azure_openai.sh <task> [--skip-keywords] [--skip-expansion]

TASK=${1:-cfa}
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SKIP_KW=false; SKIP_EXP=false
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-keywords)  SKIP_KW=true  ;;
        --skip-expansion) SKIP_EXP=true ;;
    esac; shift
done

export USE_AZURE_OPENAI=1

echo "Running pipeline for task=${TASK} (Azure OpenAI)"
[[ "$SKIP_KW"  == false ]] && python "${PROJ_DIR}/scripts/keywords_generator.py"            --task "${TASK}"
[[ "$SKIP_EXP" == false ]] && python "${PROJ_DIR}/scripts/keyword_expansion_from_corpus.py"  --task "${TASK}"
python "${PROJ_DIR}/scripts/generate_questions.py"   --task "${TASK}"
python "${PROJ_DIR}/scripts/consistency_filter.py"    --task "${TASK}"

echo "[done] pipeline finished for task=${TASK}"
