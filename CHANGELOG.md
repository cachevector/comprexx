# Changelog

All notable changes to Comprexx are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow running `pytest` on Python 3.10, 3.11, 3.12 plus a
  `ruff check` lint job.
- `CHANGELOG.md` with history for v0.1.0 and v0.2.0.

### Changed
- Silenced the noisy `torch.ao.quantization is deprecated` warning inside the
  PTQ dynamic and static stages. The underlying API is still used, with a
  `TODO(v0.3)` marking the upcoming migration to `torchao.quantization`.
- Fixed the package `__version__` to report `0.2.0` instead of the stale
  `0.1.0` that shipped on PyPI.
- Tightened the codebase against `ruff check` and added a per-file ignore
  for `E741` in tests.

## [0.2.0] - 2026-04-07

### Added
- **Unstructured pruning** stage: magnitude or random element-wise pruning
  with global/local scope and optional gradual cubic schedule.
- **N:M sparsity** stage: structured N-of-M sparsity (default 2:4) for
  NVIDIA Ampere sparse tensor cores.
- **Weight-only quantization** stage: group-wise INT4/INT8 with symmetric
  or asymmetric scaling for Linear and Conv2d layers.
- **Low-rank decomposition** stage: truncated SVD factorization of Linear
  layers, with fixed rank-ratio or energy-threshold selection modes.
- **Operator fusion** stage: Conv2d + BatchNorm2d folding via `torch.fx`
  with graceful fallback on non-traceable models.
- **Weight clustering** stage: per-layer k-means codebook clustering.
- **`cx.analyze_sensitivity()`**: per-layer sensitivity probing via prune
  or noise perturbation. Returns a `SensitivityReport` that ranks layers
  by metric drop and can suggest `exclude_layers` above a threshold.
- New techniques are wired through the recipe schema and loader, and
  exported from `comprexx.stages`.

### Tests
- 163 passing (up from 91).

## [0.1.0] - 2026-04-06

Initial release.

### Added
- Model analysis and profiling via `cx.analyze()`.
- Structured pruning with L1/L2/random criteria and global/local scope.
- Post-training dynamic and static INT8 quantization.
- ONNX export with manifest and optional `onnxruntime` validation.
- Recipe-driven pipelines (YAML) validated via Pydantic.
- CLI commands: `comprexx analyze`, `compress`, `export`.
- Accuracy guards with halt/warn actions.
- Per-stage compression reports persisted under `comprexx_runs/`.

[Unreleased]: https://github.com/cachevector/comprexx/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/cachevector/comprexx/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cachevector/comprexx/releases/tag/v0.1.0
