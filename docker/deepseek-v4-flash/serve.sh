#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash (user simulator) on one 8xH100 node as an OpenAI-compatible endpoint.
# Uses the exact image that ran for the paper, published as
# ghcr.io/compass-group-tue/figsbench-user-simulator:deepseek-v4-flash
# (image id c89bba31; originally tagged vllm-dsv4-flash:cu130-breakable). The default is pinned by
# digest, so `docker pull ghcr.io/compass-group-tue/figsbench-user-simulator:deepseek-v4-flash` and this script get the same bytes.
# vLLM args match inference-library preset deepseek-v4-flash-legacy. Do NOT set
# VLLM_USE_BREAKABLE_CUDAGRAPH=0: the image's breakable-CUDA-graph patch must stay active.
# Usage: WEIGHTS=/path/to/deepseek-ai/DeepSeek-V4-Flash PORT=8001 bash serve.sh
set -euo pipefail
: "${WEIGHTS:?set WEIGHTS to the local DeepSeek-V4-Flash checkpoint directory}"
PORT="${PORT:-8001}"
IMAGE="${IMAGE:-ghcr.io/compass-group-tue/figsbench-user-simulator@sha256:5afc74a0978b45cee8c497aee0ab74679bf22ea9898b59428e60431f996c3e4d}"
docker run -d --name "dsv4-flash-${PORT}" --gpus all --ipc=host -p "${PORT}:8000" \
  -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=1 -e NCCL_NVLS_ENABLE=1 \
  -v "${WEIGHTS}:/model" "${IMAGE}" \
  /model --tensor-parallel-size 8 --max-model-len 16384 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 16384 --block-size 256 --kv-cache-dtype fp8 \
  --enable-expert-parallel --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 --trust-remote-code \
  --optimization-level 3 --no-enable-prefix-caching -cc.cudagraph_mode=FULL_DECODE_ONLY \
  --served-model-name deepseek-v4-flash --port 8000
echo "user-simulator endpoint: http://localhost:${PORT}/v1  (export LOCAL_USER_BASE_URL=http://<host>:${PORT}/v1)"
