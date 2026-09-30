# DeepSeek-V4-Flash user simulator for H100 hosts with an older driver (e.g. 535):
# CUDA forward-compat libraries plus a breakable CUDA-graph patch.
# Build (context must be this directory): docker build -t figsbench-user-simulator:deepseek-v4-flash .
FROM vllm/vllm-openai:deepseekv4-cu130

# Fix PTX / MARLIN kernel issues when host driver is older than image CUDA (e.g. 535 vs 13.0).
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH}

COPY patches/breakable_cudagraph.py /tmp/breakable_cudagraph.py
COPY patches/apply_patches.py /tmp/apply_patches.py

RUN python3 /tmp/apply_patches.py && rm /tmp/breakable_cudagraph.py /tmp/apply_patches.py

# Sanity check: breakable module importable, DSV4 model still present.
RUN python3 -c "\
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper; \
from vllm.model_executor.models.deepseek_v4 import DeepseekV4ForCausalLM; \
print('breakable cudagraph patch OK', DeepseekV4ForCausalLM)\
"
