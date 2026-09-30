#!/usr/bin/env bash
# Serve the judge (GLM-5.3-Flash, FP8) on one 8xH100 node as an OpenAI-compatible endpoint.
# Uses the image that ran for the paper. Get it with
#   hf download compass-group-tue/FIGSBench-images figsbench-judge.tar --local-dir .
#   docker load -i figsbench-judge.tar
# These are the vLLM arguments of the paper's runs (max_model_len 98304, prefix caching on).
# Usage: WEIGHTS=/path/to/zai-org/GLM-5.3-Flash PORT=8000 bash serve.sh
set -euo pipefail
: "${WEIGHTS:?set WEIGHTS to the local GLM-5.3-Flash checkpoint directory}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-figsbench-judge:glm-5.3-flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
docker run -d --name "figsbench-judge-${PORT}" --gpus all --ipc=host -p "${PORT}:8000" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=1 -e NCCL_NVLS_ENABLE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
  -v "${WEIGHTS}:/model" "${IMAGE}" \
  /model --tensor-parallel-size 8 --max-model-len "${MAX_MODEL_LEN}" --gpu-memory-utilization 0.85 \
  --max-num-seqs "${MAX_NUM_SEQS}" --max-num-batched-tokens 32768 --kv-cache-dtype auto --enable-expert-parallel \
  --tool-call-parser glm47 --reasoning-parser glm45 --trust-remote-code --enable-prefix-caching \
  --served-model-name zai-org/GLM-5.3-Flash --port 8000
echo "judge endpoint: http://localhost:${PORT}/v1  (export LOCAL_GLM_BASE_URL=http://<host>:${PORT}/v1)"
