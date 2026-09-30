# Self-hosted models: GLM-5.3-Flash judge and DeepSeek-V4-Flash user simulator

The judge and the user simulator are open-weight models served with vLLM, each on one 8×H100
node, behind an OpenAI-compatible `/v1` endpoint. The images that ran for the paper are on
Hugging Face at [compass-group-tue/FIGSBench-images](https://huggingface.co/compass-group-tue/FIGSBench-images).
Weights are not included in the images; they are mounted at runtime.

| Role | File | Loads as | Model |
|---|---|---|---|
| Judge (`glm-5.3-flash-judge/`) | `figsbench-judge.tar` (20 GB) | `figsbench-judge:glm-5.3-flash` | `zai-org/GLM-5.3-Flash` @ `04c4e9e9` |
| User simulator (`deepseek-v4-flash/`) | `figsbench-user-simulator.tar` (23 GB) | `figsbench-user-simulator:deepseek-v4-flash` | `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d707` |

Use the original DeepSeek-V4-Flash release, not `DeepSeek-V4-Flash-0731`.

In the code, the judge is `local-glm-5.3-flash` (endpoint `LOCAL_GLM_BASE_URL`) and the user
simulator is `local-deepseek-v4-flash` (endpoint `LOCAL_USER_BASE_URL`). Any other model name goes
to OpenRouter.

## Get the images

```bash
hf download compass-group-tue/FIGSBench-images --local-dir ./images
docker load -i images/figsbench-judge.tar
docker load -i images/figsbench-user-simulator.tar
```

## Why two images

The judge and the simulator ran on different vLLM builds. Serving either model on the other
model's stack would be a configuration the paper never measured, so the release keeps them apart:

| | Judge image | User-simulator image |
|---|---|---|
| vLLM | `0.1.dev20051+g487ecf187` | `0.1.dev15833+g62d441ee8` plus the breakable-CUDA-graph patch |
| torch | 2.13.0+cu130 | 2.11.0+cu130 |
| transformers | 5.15.1 | 4.57.6 |
| flashinfer | 0.6.17 | 0.6.8.post1 |
| NCCL | 2.30.7 | 2.28.9 |
| CUDA | 13.0.1 | 13.0.1 |

## Build from the recipes

```bash
docker build -t figsbench-judge:glm-5.3-flash glm-5.3-flash-judge
docker build -t figsbench-user-simulator:deepseek-v4-flash deepseek-v4-flash
```

* `glm-5.3-flash-judge/Dockerfile` starts from `vllm/vllm-openai:glm53-flash`. It adds only CUDA
  forward-compat libraries and environment settings.
* `deepseek-v4-flash/Dockerfile` starts from `vllm/vllm-openai:deepseekv4-cu130` and applies `apply_patches.py`, which
  installs a breakable CUDA-graph wrapper for `DeepseekV4ForCausalLM` so the model runs on
  hosts whose driver predates the image's CUDA 13.0 (the scripts are in `deepseek-v4-flash/patches/`).
  The build finishes with an import check.

Rebuilds pull the upstream `vllm/vllm-openai` base tags, which may have moved since the paper's
runs. For comparable numbers, use the published images above; the recipes document how they were
made. (The published judge image also has different label text and one unused environment
variable; neither affects serving.)

## Serve

```bash
WEIGHTS=/path/to/GLM-5.3-Flash      PORT=8000 bash glm-5.3-flash-judge/serve.sh
WEIGHTS=/path/to/DeepSeek-V4-Flash  PORT=8001 bash deepseek-v4-flash/serve.sh

export LOCAL_GLM_BASE_URL=http://<judge-host>:8000/v1
export LOCAL_USER_BASE_URL=http://<user-host>:8001/v1
```

The vLLM arguments in each `serve.sh` are the ones the paper's servers ran with (also listed in
`serve_presets/*.yaml`). Key settings:

* **GLM-5.3-Flash:** TP 8 with expert parallelism, `max_model_len 98304`, `max_num_seqs 64`,
  `max_num_batched_tokens 32768`, KV cache `auto` (BF16 on Hopper), `--enable-prefix-caching`,
  `--reasoning-parser glm45`, `--tool-call-parser glm47`, served as `zai-org/GLM-5.3-Flash`.
* **DeepSeek-V4-Flash:** TP 8 with expert parallelism, `max_model_len 16384`, FP8 KV cache,
  `block-size 256`, `--optimization-level 3`, `-cc.cudagraph_mode=FULL_DECODE_ONLY`, prefix
  caching off. Do **not** set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`: the image's breakable
  CUDA-graph patch must stay active.

## Judge request settings

These are set by the client code (`src/figsbench/judge/stages.py`), not by the server:

* Set `JUDGE_MAX_TOKENS=64000`. Hard samples need more than 32k tokens of reasoning, and smaller
  budgets can return empty content.
* Judge timeout is 900 s per request.
* The OpenRouter judge (`--judge api`) defaults to `google/gemini-3.8-flash`; use
  `--judge-model z-ai/glm-5.3-flash` for GLM through OpenRouter. All reported judgments used the
  local endpoint (`judge_model: local-glm-5.3-flash` in every result file). Results from any
  other judge are labelled `paper_judge: false`.
