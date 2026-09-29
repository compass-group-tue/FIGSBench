FROM vllm/vllm-openai:glm53-flash
LABEL org.opencontainers.image.source="https://github.com/psidharth567/toolkit" \
    org.opencontainers.image.title="toolkit/inference-glm53" \
    org.opencontainers.image.description="Inference GLM-5.3-Flash cu130 32768" \
    org.opencontainers.image.base.name="vllm/vllm-openai:glm53-flash"

ENV DEBIAN_FRONTEND=noninteractive

ENV LD_LIBRARY_PATH=/usr/local/cuda/compat:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    NCCL_NVLS_ENABLE=1 \
    VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=64 \
    VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    VLLM_ALLREDUCE_USE_FLASHINFER=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=0 \
    TOOLKIT_WEIGHTS=/weights \
    HF_HOME=/cache

USER root
RUN apt-get update && apt-get install -y --no-install-recommends cuda-compat-12-8 2>&1 | tail -5 || echo "cuda-compat-12-8 not available in this base, using image compat" && \
    rm -rf /var/lib/apt/lists/* || true

RUN python3 -c "import vllm; print('vllm', vllm.__version__)" && \
    ls -lh /usr/local/cuda/compat/ 2>&1 | head -20 || true

WORKDIR /workspace
CMD ["bash"]
