# Self-hosted models: GLM-5.3-Flash judge and DeepSeek-V4-Flash user simulator

Two open-weight models are served locally with vLLM, each on one 8×H100 node, behind an
OpenAI-compatible `/v1` endpoint. The exact images used for the paper are published as GitHub
packages of this repository, so serving needs no other repository or registry. Weights are never
baked in and are bind-mounted at runtime.

| Role | Image (pinned by digest in `serve.sh`) | Image id | Originally tagged |
|---|---|---|---|
| **Judge** (`glm-5.3-flash-judge/`) | `ghcr.io/compass-group-tue/figsbench-judge:glm-5.3-flash`<br>`@sha256:9fc5cb280ecd19cf57d4d2feb42ffad35930193bf4852e8d53d1a3209c757a7c` | `sha256:6a5c3471f9199628a9da84d224599a84267bd7c53e636661d3c59e620d437e03` | `toolkit/inference-glm53:12.8` |
| **User simulator** (`deepseek-v4-flash/`) | `ghcr.io/compass-group-tue/figsbench-user-simulator:deepseek-v4-flash`<br>`@sha256:5afc74a0978b45cee8c497aee0ab74679bf22ea9898b59428e60431f996c3e4d` | `sha256:c89bba31fc32572db8cb1ae742bc0a04b1d3d0a33e516bf5cfa34d8929b2422a` | `vllm-dsv4-flash:cu130-breakable` |

These are the `legacy` images (inference-library presets `glm-5.3-flash-legacy` and
`deepseek-v4-flash-legacy`) that served every run in the paper: authoring, the difficulty gate,
scenario expansion and all model evaluations. The image id is the digest of the image config, and
it is identical to the id of the image that ran for the paper. The registry copies were pushed from `docker save` exports of those images with
their layers gzip-compressed, which changes the manifest digest but not the image id (download is
about 9.5 GB for each image).

| Role | Model | Client alias in code | Endpoint env var | Directory |
|---|---|---|---|---|
| **Judge.** All four hr-v11 stages, used for every reported score. Also the scenario-expansion rewriter. | `zai-org/GLM-5.3-Flash` (FP8) @ `04c4e9e9` | `local-glm-5.3-flash` | `LOCAL_GLM_BASE_URL`, `LOCAL_GLM_MODEL` | `glm-5.3-flash-judge/` |
| **User simulator.** Plays the user in authoring and in every evaluation rollout. | `deepseek-ai/DeepSeek-V4-Flash` @ `60d8d707` (the 0423 release, **not** `-0731`) | `local-deepseek-v4-flash` | `LOCAL_USER_BASE_URL`, `LOCAL_USER_MODEL` | `deepseek-v4-flash/` |

The routing lives in `code/benchmark/pipeline/client.py` (`LOCAL_MODEL_ROUTES`). Any model name
starting with `local-` is sent to the endpoint named by the env var; everything else goes to OpenRouter.

## Pull

```bash
docker pull ghcr.io/compass-group-tue/figsbench-judge@sha256:9fc5cb280ecd19cf57d4d2feb42ffad35930193bf4852e8d53d1a3209c757a7c
docker pull ghcr.io/compass-group-tue/figsbench-user-simulator@sha256:5afc74a0978b45cee8c497aee0ab74679bf22ea9898b59428e60431f996c3e4d
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

## Build (reproducible path)

```bash
# GLM-5.3-Flash judge (recipe byte-identical to the toolkit file that built the published image)
docker build -t toolkit/inference-glm53:12.8 -f glm-5.3-flash-judge/Dockerfile glm-5.3-flash-judge

# DeepSeek-V4-Flash: the build context must be deepseek-v4-flash/ because the Dockerfile COPYs
# inference-library/docker/dsv4-flash/{apply_patches.py,breakable_cudagraph.py}
docker build -t vllm-dsv4-flash:cu130-breakable -f deepseek-v4-flash/Dockerfile deepseek-v4-flash
```

* `glm-5.3-flash-judge/Dockerfile` starts from `vllm/vllm-openai:glm53-flash`. It adds only CUDA
  forward-compat libraries and environment settings.
* `deepseek-v4-flash/Dockerfile` is the legacy `dsv4-flash-legacy` recipe that built the published image:
  it starts from `vllm/vllm-openai:deepseekv4-cu130` and applies `apply_patches.py`, which
  installs a breakable CUDA-graph wrapper for `DeepseekV4ForCausalLM` so the model runs on
  hosts whose driver predates the image's CUDA 13.0. The build finishes with an import check.

Rebuilds pull the upstream `vllm/vllm-openai` base tags, which may have moved since the paper's
runs. For comparable numbers, use the published images above; the recipes document how they were
made.

## Serve

```bash
WEIGHTS=/path/to/GLM-5.3-Flash      PORT=8000 bash glm-5.3-flash-judge/serve.sh
WEIGHTS=/path/to/DeepSeek-V4-Flash  PORT=8001 bash deepseek-v4-flash/serve.sh

export LOCAL_GLM_BASE_URL=http://<judge-host>:8000/v1
export LOCAL_USER_BASE_URL=http://<user-host>:8001/v1
```

The vLLM arguments in each `serve.sh` match the exact servers used for this release
(`serve_presets/*.yaml`). The `docker run` flags (`--gpus all --ipc=host`, weights mounted
at `/model`) match the toolkit launcher. Key settings:

* **GLM-5.3-Flash:** TP 8 with expert parallelism, `max_model_len 98304`, `max_num_seqs 64`,
  `max_num_batched_tokens 32768`, KV cache `auto` (BF16 on Hopper), `--enable-prefix-caching`,
  `--reasoning-parser glm45`, `--tool-call-parser glm47`, served as `zai-org/GLM-5.3-Flash`.
* **DeepSeek-V4-Flash:** TP 8 with expert parallelism, `max_model_len 16384`, FP8 KV cache,
  `block-size 256`, `--optimization-level 3`, `-cc.cudagraph_mode=FULL_DECODE_ONLY`, prefix
  caching off. Do **not** set `VLLM_USE_BREAKABLE_CUDAGRAPH=0`: the image's breakable
  CUDA-graph patch must stay active.

## Judge request settings

These are set by `code/rule_guided_judging` and `code/benchmark/pipeline/client.py`, not by the
server:

* Set `JUDGE_MAX_TOKENS=64000`. Hard samples need more than 32k tokens of reasoning, and smaller
  budgets can return empty content.
* Judge timeout is 900 s per request.
* The OpenRouter judge (`--judge api`) defaults to `google/gemini-3.8-flash`; use
  `--judge-model z-ai/glm-5.3-flash` for GLM through OpenRouter. All reported judgments used the
  local endpoint (`judge_model: local-glm-5.3-flash` in every result file), and only that judge
  gives `paper_judge: true`.
