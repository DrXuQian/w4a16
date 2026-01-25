# w4a16

This repo is a scratchpad to compare/extract W4A16 kernels (FP16 activations, INT4 weights).

Included projects
-----------------
- `fpA_intB_standalone/`: TensorRT-LLM extracted fpA_intB (FP16 x INT4, GPTQ scale+zero)
  - CUTLASS path + CUDA fallback for small M (e.g. M=1)
  - TRT-LLM-style config profiling (`--list_configs`, `--config=...`, `FPA_INTB_PROFILE_LOG=1`)
  - Helper script to run all configs directly: `scripts/run_all_configs.sh`
- `marlin_sm110_kernel/`: vLLM extracted Marlin INT4 GPTQ kernel + selector + standalone CLI
- `homemade_marlin/`: standalone Marlin kernel variants used for performance comparison
  - `MARLIN_USE_CREATEPOLICY=0` by default to avoid `createpolicy` illegal-instruction issues on some setups

Notes
-----
- Build directories are intentionally ignored (see `.gitignore`).
- Each subproject defaults to a target arch in its own `CMakeLists.txt`; override with
  `-DCMAKE_CUDA_ARCHITECTURES=...` when building on a different GPU.
