#!/usr/bin/env python3
"""Patch vLLM in the deepseekv4-cu130 image with breakable CUDA graph support."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")
PATCH_MARKER = "# PATCHED: breakable_cudagraph"


def _read(path: Path) -> str:
    return path.read_text()


def _write(path: Path, content: str) -> None:
    path.write_text(content)
    print(f"patched {path}")


def _replace_once(path: Path, old: str, new: str, label: str) -> None:
    content = _read(path)
    if PATCH_MARKER in content and old not in content:
        print(f"skip {label}: already patched")
        return
    if old not in content:
        raise RuntimeError(f"patch anchor not found for {label} in {path}")
    _write(path, content.replace(old, new, 1))


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    breakable_src = script_dir / "breakable_cudagraph.py"
    breakable_dst = VLLM_ROOT / "compilation" / "breakable_cudagraph.py"
    shutil.copy2(breakable_src, breakable_dst)
    print(f"installed {breakable_dst}")

    envs_path = VLLM_ROOT / "envs.py"
    envs = _read(envs_path)
    if "VLLM_USE_BREAKABLE_CUDAGRAPH" not in envs:
        envs = envs.replace(
            "    VLLM_ENABLE_PREGRAD_PASSES: bool = False\n",
            "    VLLM_ENABLE_PREGRAD_PASSES: bool = False\n"
            "    VLLM_USE_BREAKABLE_CUDAGRAPH: bool = False\n",
            1,
        )
        envs = envs.replace(
            '    "VLLM_ENABLE_PREGRAD_PASSES": lambda: (\n'
            '        os.environ.get("VLLM_ENABLE_PREGRAD_PASSES", "0") == "1"\n'
            "    ),\n",
            '    "VLLM_ENABLE_PREGRAD_PASSES": lambda: (\n'
            '        os.environ.get("VLLM_ENABLE_PREGRAD_PASSES", "0") == "1"\n'
            "    ),\n"
            '    "VLLM_USE_BREAKABLE_CUDAGRAPH": lambda: (\n'
            '        os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH", "0") == "1"\n'
            "    ),\n",
            1,
        )
        _write(envs_path, envs)
    else:
        print("skip envs.py: already patched")

    config_path = VLLM_ROOT / "config" / "vllm.py"
    _replace_once(
        config_path,
        """            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        if self.compilation_config.backend == "eager" or (
            self.compilation_config.mode is not None
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        def has_blocked_weights():""",
        f"""            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE

        {PATCH_MARKER}
        if (
            self.model_config is not None
            and "VLLM_USE_BREAKABLE_CUDAGRAPH" not in os.environ
            and any(
                a in ("DeepseekV4ForCausalLM", "DeepSeekV4MTPModel")
                for a in self.model_config.architectures
            )
        ):
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
            logger.info_once(
                "Auto-enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1 for DeepSeek V4. "
                "Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 to opt out."
            )

        if envs.VLLM_USE_BREAKABLE_CUDAGRAPH:
            logger.warning_once(
                "VLLM_USE_BREAKABLE_CUDAGRAPH is set, disabling vLLM's "
                "torch.compile pipeline. Equivalent to -cc.mode=none."
            )
            self.compilation_config.mode = CompilationMode.NONE

        if self.compilation_config.backend == "eager" or (
            self.compilation_config.mode is not None
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.warning(
                "Inductor compilation was disabled by user settings, "
                "optimizations settings that are only active during "
                "inductor compilation will be ignored."
            )

        def has_blocked_weights():""",
        "config/vllm.py auto-enable",
    )

    _replace_once(
        config_path,
        """        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
        ):
            logger.info(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE""",
        """        if (
            self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            and self.compilation_config.mode != CompilationMode.VLLM_COMPILE
            and not envs.VLLM_USE_BREAKABLE_CUDAGRAPH
        ):
            logger.info(
                "Cudagraph mode %s is not compatible with compilation mode %s."
                "Overriding to NONE.",
                self.compilation_config.cudagraph_mode,
                self.compilation_config.mode,
            )
            self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE""",
        "config/vllm.py piecewise bypass",
    )

    _replace_once(
        config_path,
        """            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert self.compilation_config.mode == CompilationMode.VLLM_COMPILE, (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )""",
        """            if self.compilation_config.cudagraph_mode.requires_piecewise_compilation():
                assert (
                    self.compilation_config.mode == CompilationMode.VLLM_COMPILE
                    or envs.VLLM_USE_BREAKABLE_CUDAGRAPH
                ), (
                    "Compilation mode should be CompilationMode.VLLM_COMPILE "
                    "when cudagraph_mode piecewise cudagraphs is used, "
                    f"cudagraph_mode={self.compilation_config.cudagraph_mode}"
                )""",
        "config/vllm.py assert bypass",
    )

    runner_path = VLLM_ROOT / "v1" / "worker" / "gpu_model_runner.py"
    _replace_once(
        runner_path,
        "from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper\n",
        "from vllm.compilation.breakable_cudagraph import (\n"
        "    BreakableCUDAGraphWrapper,\n"
        "    is_breakable_cudagraph_enabled,\n"
        ")\n"
        "from vllm.compilation.cuda_graph import CUDAGraphStat, CUDAGraphWrapper\n",
        "gpu_model_runner imports",
    )

    _replace_once(
        runner_path,
        """        if (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:""",
        """        if (
            is_breakable_cudagraph_enabled()
            and cudagraph_mode != CUDAGraphMode.NONE
            and not self.parallel_config.use_ubatching
        ):
            self.model = BreakableCUDAGraphWrapper(self.model, self.vllm_config)
            drafter = getattr(self, "drafter", None)
            if drafter is not None and hasattr(drafter, "model"):
                drafter.model = BreakableCUDAGraphWrapper(
                    drafter.model, self.vllm_config
                )
        elif (
            cudagraph_mode.has_full_cudagraphs()
            and not self.parallel_config.use_ubatching
        ):
            self.model = CUDAGraphWrapper(
                self.model, self.vllm_config, runtime_mode=CUDAGraphMode.FULL
            )
        elif self.parallel_config.use_ubatching:""",
        "gpu_model_runner wrapper selection",
    )

    _replace_once(
        runner_path,
        "        if isinstance(self.model, (CUDAGraphWrapper, UBatchWrapper)):\n",
        "        if isinstance(\n"
        "            self.model, (CUDAGraphWrapper, UBatchWrapper, BreakableCUDAGraphWrapper)\n"
        "        ):\n",
        "gpu_model_runner get_model",
    )

    _replace_once(
        runner_path,
        "        for instance in list(CUDAGraphWrapper._all_instances):\n"
        "            original_pools[id(instance)] = instance.graph_pool\n"
        "            instance.graph_pool = profiling_pool\n",
        "        all_wrappers = list(CUDAGraphWrapper._all_instances) + list(\n"
        "            BreakableCUDAGraphWrapper._all_instances\n"
        "        )\n"
        "        for instance in all_wrappers:\n"
        "            original_pools[id(instance)] = instance.graph_pool\n"
        "            instance.graph_pool = profiling_pool\n",
        "gpu_model_runner profiling pool",
    )

    _replace_once(
        runner_path,
        "        CUDAGraphWrapper.clear_all_graphs()\n"
        "        for instance in list(CUDAGraphWrapper._all_instances):\n",
        "        CUDAGraphWrapper.clear_all_graphs()\n"
        "        BreakableCUDAGraphWrapper.clear_all_graphs()\n"
        "        for instance in list(CUDAGraphWrapper._all_instances) + list(\n"
        "            BreakableCUDAGraphWrapper._all_instances\n"
        "        ):\n",
        "gpu_model_runner clear graphs",
    )

    dispatcher_path = VLLM_ROOT / "v1" / "cudagraph_dispatcher.py"
    _replace_once(
        dispatcher_path,
        """        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )""",
        """        from vllm.compilation.breakable_cudagraph import (
            is_breakable_cudagraph_enabled,
        )

        assert (
            not self.compilation_config.cudagraph_mode.requires_piecewise_compilation()
            or self.compilation_config.is_attention_compiled_piecewise()
            or is_breakable_cudagraph_enabled()
        ), (
            "Compilation mode should be CompilationMode.VLLM_COMPILE when "
            "cudagraph_mode piecewise cudagraphs is used, "
            "and attention should be in splitting_ops or "
            "inductor splitting should be used. "
            f"cudagraph_mode={self.compilation_config.cudagraph_mode}, "
            f"compilation_mode={self.compilation_config.mode}, "
            f"splitting_ops={self.compilation_config.splitting_ops}"
        )""",
        "cudagraph_dispatcher assert",
    )

    attn_path = VLLM_ROOT / "model_executor" / "layers" / "deepseek_v4_attention.py"
    _replace_once(
        attn_path,
        "from vllm.utils.torch_utils import direct_register_custom_op\n",
        "from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n"
        "from vllm.utils.torch_utils import direct_register_custom_op\n",
        "deepseek_v4_attention import",
    )
    _replace_once(
        attn_path,
        "def deepseek_v4_attention(\n",
        "@eager_break_during_capture\ndef deepseek_v4_attention(\n",
        "deepseek_v4_attention decorator",
    )

    indexer_path = VLLM_ROOT / "model_executor" / "layers" / "sparse_attn_indexer.py"
    _replace_once(
        indexer_path,
        "import vllm.envs as envs\n",
        "import vllm.envs as envs\n"
        "from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n",
        "sparse_attn_indexer import",
    )
    _replace_once(
        indexer_path,
        "def sparse_attn_indexer(\n",
        "@eager_break_during_capture\ndef sparse_attn_indexer(\n",
        "sparse_attn_indexer decorator",
    )

    print("all patches applied successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
