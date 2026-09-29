# Recipe for the legacy local image vllm-dsv4-flash:cu130-breakable (sha256:c89bba31fc32), used by
# inference-library preset deepseek-v4-flash-legacy. Build: docker build -t vllm-dsv4-flash:cu130-breakable deepseek-v4-flash
# DeepSeek V4 Flash on H100 with driver 535: CUDA forward-compat + breakable cudagraph.
FROM vllm/vllm-openai:deepseekv4-cu130

# Fix PTX / MARLIN kernel issues when host driver is older than image CUDA (e.g. 535 vs 13.0).
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}

COPY inference-library/docker/dsv4-flash/breakable_cudagraph.py /tmp/breakable_cudagraph.py
COPY inference-library/docker/dsv4-flash/apply_patches.py /tmp/apply_patches.py

RUN python3 /tmp/apply_patches.py && rm /tmp/breakable_cudagraph.py /tmp/apply_patches.py

# Sanity check: breakable module importable, DSV4 model still present.
RUN python3 -c "\
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper; \
from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM; \
print('breakable cudagraph patch OK', DeepseekV4ForCausalLM)\
"
