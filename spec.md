# Comprexx — Specification Document

> **"Compress smarter. Ship faster. Run anywhere."**
>
> Comprexx is an open-source, developer-first ML model compression platform that turns
> bloated research models into production-ready artifacts — with full observability into
> the accuracy/size/latency trade-off at every step of the pipeline.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [The Problem Worth Solving](#2-the-problem-worth-solving)
3. [What Comprexx Is — and Is Not](#3-what-comprexx-is--and-is-not)
4. [Target Users](#4-target-users)
5. [Core Philosophy & Design Principles](#5-core-philosophy--design-principles)
6. [Feature Set](#6-feature-set)
   - 6.1 Model Ingestion & Analysis Engine
   - 6.2 Compression Pipeline
   - 6.3 Compression Techniques
   - 6.4 The Recipe System
   - 6.5 Accuracy Recovery
   - 6.6 Export & Deployment Targets
   - 6.7 Visualization & Observability Dashboard
   - 6.8 CLI Interface
   - 6.9 Python SDK API
   - 6.10 CI/CD & Automation Integration
   - 6.11 Good-to-Haves (v2+)
7. [Architecture](#7-architecture)
8. [Tech Stack](#8-tech-stack)
9. [The Recipe DSL (Detailed)](#9-the-recipe-dsl-detailed)
10. [Visualization Specs](#10-visualization-specs)
11. [CLI Specification](#11-cli-specification)
12. [Python SDK Specification](#12-python-sdk-specification)
13. [Export Targets Specification](#13-export-targets-specification)
14. [Benchmark & Evaluation Framework](#14-benchmark--evaluation-framework)
15. [Configuration & Persistence](#15-configuration--persistence)
16. [Error Handling & Developer Experience](#16-error-handling--developer-experience)
17. [Testing Strategy](#17-testing-strategy)
18. [Repository Structure](#18-repository-structure)
19. [Roadmap & Versioning](#19-roadmap--versioning)
20. [Competitive Positioning](#20-competitive-positioning)

---

## 1. Project Identity

| Field        | Value |
|--------------|-------|
| **Name**     | Comprexx |
| **Tagline**  | Compress smarter. Ship faster. Run anywhere. |
| **Category** | ML Model Compression & Deployment Optimization Toolkit |
| **Positioning** | The missing middle layer between training and deployment — a unified, observable, composable compression pipeline for serious ML engineers |
| **License**  | Apache 2.0 |
| **Primary Language** | Python 3.10+ |
| **Target Audience** | ML Engineers, MLOps Engineers, Edge AI Developers, Researchers who ship |

---

## 2. The Problem Worth Solving

### The Compression Ecosystem is Broken for Practitioners

The model compression space in 2025 is rich in academic techniques and fragmented in tooling.
The typical workflow for an ML engineer who wants to compress and ship a model looks like:

1. Find a pruning script on GitHub — likely a research repo with hard-coded paths
2. Find a separate quantization library (bitsandbytes, AutoGPTQ, AutoAWQ, torch.quantization)
3. Write glue code to chain them
4. Discover none of these export cleanly to ONNX
5. Rebuild the export layer
6. Realize there's no way to compare accuracy across compression levels systematically
7. Repeat with a different combination

**Neural Magic's LLM Compressor** addresses this for vLLM/HuggingFace transformers.
**TensorRT** addresses this for NVIDIA hardware specifically.
**TorchScript/ONNX** address export, not compression.

**No tool does all of the following in one place:**
- Model-agnostic compression (CNNs, Transformers, custom architectures)
- Composable multi-technique pipelines (prune → quantize → distill → export)
- First-class observability (what exactly changed, and what did it cost)
- Hardware-aware recommendations (given your target device, here's what to do)
- A recipe system that makes compression reproducible and shareable
- An interactive visualization layer that communicates trade-offs clearly

Comprexx fills this gap — not as a research tool but as an **engineering product**.

---

## 3. What Comprexx Is — and Is Not

### What it IS:
- A **compression pipeline orchestrator** — chain techniques in a defined, reproducible order
- A **trade-off explorer** — visualize size, FLOPs, latency, and accuracy impact at every level
- A **hardware-aware advisor** — recommend compression strategies given target hardware
- A **recipe-driven system** — define compression as code, version it, share it
- A **multi-format exporter** — PyTorch, ONNX, TorchScript, TFLite, TensorRT
- A **Python-first SDK** with a strong CLI for CI/CD integration

### What it is NOT:
- A training framework (no custom training loops — integrate with yours)
- A model serving layer (export and hand off to vLLM, TorchServe, ONNX Runtime, etc.)
- An AutoML tool (no architecture search — it works on models you already have)
- LLM-only (works on any PyTorch model, not just transformers)

---

## 4. Target Users

### Primary: ML Engineer / Applied Researcher
Trains models, needs to get them onto edge devices or reduce cloud inference costs.
Wants a programmable, composable API. Comfortable writing Python.
Pain: Stitching together 5 different libraries to do what should be one workflow.

### Secondary: MLOps / Infrastructure Engineer
Manages the model lifecycle from training to serving.
Needs compression baked into CI/CD pipelines.
Pain: No standardized artifact format or reproducibility story for compressed models.

### Tertiary: Edge AI Developer
Building for Jetson, Raspberry Pi, WASM, mobile.
Needs to hit specific size/latency budgets.
Pain: No tool tells them *which combination* of techniques gets them to their target without brute-force experimentation.

### Out of scope (v1):
- Non-technical users / GUI-only users
- Data scientists who don't write code
- Teams wanting a managed cloud service (potential v3 consideration)

---

## 5. Core Philosophy & Design Principles

### 1. Recipes over scripts
Compression is reproducible, versionable, and shareable configuration — not a pile of one-off scripts.
A Comprexx recipe is a YAML/Python object that fully describes a compression pipeline.

### 2. Observability is not optional
Every compression step must report what it changed and what it cost in accuracy, size, and latency.
Compression without measurement is engineering malpractice. Comprexx makes measurement the default.

### 3. Composability
Techniques are modules. Pipelines are sequences of modules.
Any technique can be composed with any other without custom glue code.

### 4. Hardware-aware by default
Compression decisions are not made in a vacuum. Comprexx asks: "What are you running this on?"
and uses that to constrain and recommend the strategy.

### 5. Fail loudly, recover gracefully
If accuracy drops past a configured threshold, the pipeline halts and tells the user exactly why.
Recovery techniques (distillation, QAT) are available as pipeline stages.

### 6. Developer experience is a feature
Meaningful error messages. Typed Python API. Auto-complete friendly.
No undocumented magic. No "it works on my machine."

---

## 6. Feature Set

---

### 6.1 Model Ingestion & Analysis Engine

Before any compression, Comprexx analyzes the model and produces a **model profile**.

**Capabilities:**
- Load any `nn.Module` from PyTorch checkpoint, HuggingFace Hub, or local path
- Load ONNX models for analysis (read-only; compression requires PyTorch)
- Auto-detect model architecture category:
  - CNN (ResNet, EfficientNet, MobileNet, custom)
  - Transformer (BERT, ViT, LLaMA, GPT-style, custom)
  - Hybrid
  - RNN/LSTM
  - Unknown (manual profile required)
- Compute and report:
  - Total parameter count (trainable + frozen)
  - Parameter count per layer/block
  - Model size on disk (FP32 baseline, FP16 estimate)
  - FLOPs / MACs (floating-point operations, multiply-accumulate operations)
  - Memory footprint at inference (peak activation memory estimate)
  - Layer-by-layer breakdown (type, shape, parameter count, FLOPs)
- Identify compressible vs. non-compressible layers automatically
- Flag layers known to be sensitive to compression (e.g., attention layers, batch norm, final classifiers)
- Produce a **compression opportunity score** per layer (heuristic-based)
- Generate a **model profile JSON** that is the basis for recipe recommendations

**Implementation Notes:**
- FLOPs measured via `torch.fx` tracing + custom FLOPs counter hooks
- Alternatively use `fvcore.nn.FlopCountAnalysis` as fallback
- Memory estimation via `torch.cuda.memory_allocated` hooks during a forward pass

---

### 6.2 Compression Pipeline

The core abstraction. A pipeline is an ordered sequence of compression stages applied to a model.

**Pipeline semantics:**
- Each stage receives a model and returns a compressed model + stage report
- Stages are stateless transforms (no side effects outside the model)
- The pipeline tracks the model state at each stage (checkpointing optional)
- Each stage emits a `StageReport`: parameter delta, size delta, FLOPs delta, accuracy delta (if eval data provided)
- The full pipeline emits a `CompressionReport` aggregating all stage reports

**Pipeline configuration:**
- Defined via Python API or YAML recipe
- Stages are executed in order
- Each stage has configurable hyperparameters
- Accuracy guard rails: if accuracy drop exceeds configured threshold at any stage, pipeline halts

**Execution modes:**
- `dry_run`: Profile and estimate only — no actual compression applied
- `apply`: Run the full pipeline
- `interactive`: Pause after each stage for inspection (useful in notebooks)

---

### 6.3 Compression Techniques

All techniques are implemented as `CompressionStage` subclasses with a consistent interface.

---

#### A. Quantization

**Post-Training Quantization (PTQ)**
- Dynamic quantization (weights only; activations quantized at runtime)
  - Targets: Linear, LSTM layers
  - Formats: INT8
- Static quantization (weights + activations; requires calibration data)
  - Calibration: min/max, percentile, entropy, OMSE
  - Formats: INT8, INT4 (via custom backend)
- Weight-only quantization
  - Formats: INT8 (W8A16), INT4 (W4A16), NF4 (for LoRA-style)
  - Algorithms: Round-to-Nearest (RTN), GPTQ, AWQ
- Mixed-precision quantization
  - Assign different precision per layer based on sensitivity analysis
  - Sensitivity computed via Hessian approximation or output perturbation

**Quantization-Aware Training (QAT)**
- Insert fake-quantization nodes into the model graph
- Produces a QAT-ready model the user can fine-tune
- Compatible with standard PyTorch training loops
- Supported formats: INT8

**Formats supported:**
- FP32 → FP16
- FP32/FP16 → INT8
- FP32/FP16 → INT4
- FP32/FP16 → NF4
- FP32/FP16 → FP8 (where hardware supports)

**Backend dispatch:**
- torch.quantization (native PTQ/QAT)
- bitsandbytes (INT8/NF4, GPU-focused)
- GPTQ via auto-gptq (for transformer weight-only quant)
- Custom INT4 kernels via torch.fx graph rewriting (where applicable)

---

#### B. Pruning

**Unstructured Pruning**
- Magnitude-based (L1, L2 norm of weights)
- Random baseline
- Gradual magnitude pruning (GMP) schedule
- Output: sparse weight tensors

**Structured Pruning**
- Filter/channel pruning (removes entire convolutional filters)
  - Criteria: L1 norm, activation statistics, Taylor expansion
- Attention head pruning (for Transformers)
  - Criteria: attention entropy, head importance scoring
- Layer pruning (remove entire layers from residual networks)
  - Criteria: block sensitivity analysis
- Output: structurally smaller model (no sparsity, direct speedup on any hardware)

**Semi-structured / N:M Sparsity**
- 2:4 sparsity for NVIDIA Ampere+ (native hardware acceleration)
- 4:8 patterns (custom)

**Pruning Schedules:**
- One-shot (prune all at once)
- Iterative (prune → fine-tune → prune cycles)
- Gradual (linearly increase sparsity over training steps — requires user's training loop integration)

**Global vs. Local:**
- Global: sparsity budget across entire model
- Local: per-layer sparsity targets

---

#### C. Knowledge Distillation

Comprexx treats distillation as a compression **recovery** stage more than a standalone technique.
It produces a student model configuration and a training recipe, but requires the user's training setup.

**Modes:**
- Response-based distillation (soft targets from teacher's logits)
- Feature-based distillation (intermediate layer activations)
- Relation-based distillation (pairwise layer relationship matching)

**What Comprexx provides:**
- Student model configuration (pruned/smaller architecture based on the teacher)
- Distillation loss functions (`KLDivLoss` with temperature, `MSELoss` for features)
- `DistillationTrainer` wrapper (drop-in replacement for a standard PyTorch training loop)
- Teacher model hooks (intercept intermediate activations)

**What the user provides:**
- Training data + DataLoader
- Training loop or HuggingFace Trainer configuration

---

#### D. Low-Rank Decomposition (LRD)

- Decompose weight matrices via Singular Value Decomposition (SVD)
- Replace a layer `W (m×n)` with two smaller matrices `U (m×r)` and `V (r×n)` where `r << min(m,n)`
- Rank selection: automatic (energy threshold), manual (explicit rank or rank ratio)
- Applied to: Linear layers, attention projection matrices (Q/K/V/O), embedding layers
- Compression ratio vs. accuracy trade-off computed per layer
- Particularly effective on Transformer attention blocks

---

#### E. Operator Fusion (Graph Optimization)

- Identify fusable operation patterns in the computation graph
- Fuse Conv+BN+ReLU → single fused op
- Fuse Linear+GeLU, Linear+LayerNorm where supported
- Eliminates intermediate activations → reduces memory bandwidth
- Implemented via `torch.fx` graph transformation passes
- ONNX-level fusion for export (via ONNX graph optimizer)

---

#### F. Weight Sharing / Clustering

- K-means clustering of weights into N centroids
- Replace individual weights with centroid indices + codebook
- Reduces unique weight values → compresses via lookup table
- Configurable: number of clusters, scope (per-layer, per-tensor)

---

### 6.4 The Recipe System

A Recipe is a first-class, serializable object that describes a complete compression pipeline.

**Recipe formats:**
- Python object (native, full type safety)
- YAML file (human-readable, version-controllable)
- JSON (machine-readable, API-friendly)

**Recipe structure (YAML example):**

```yaml
name: resnet50-edge-optimized
version: "1.0"
description: "ResNet-50 compressed for Jetson Nano deployment"
target_hardware: jetson_nano
base_model: torchvision.models.resnet50

accuracy_guard:
  metric: top1_accuracy
  baseline_threshold: 0.01  # Halt if accuracy drops more than 1%
  evaluation_dataset: imagenet_val_subset

stages:
  - name: filter_pruning
    technique: structured_pruning
    config:
      target: filters
      sparsity: 0.30
      criteria: l1_norm
      scope: global
      exclude_layers: ["layer4.1.conv2", "fc"]

  - name: quantization
    technique: ptq_static
    config:
      format: int8
      calibration_method: percentile
      calibration_samples: 512
      exclude_layers: ["fc"]

  - name: graph_optimization
    technique: operator_fusion
    config:
      patterns: ["conv_bn_relu", "linear_gelu"]

export:
  formats: ["pytorch", "onnx", "tensorrt"]
  onnx_opset: 17
  tensorrt_precision: int8
  output_dir: ./artifacts/resnet50-edge
```

**Recipe features:**
- `comprexx recipe validate` — validate a recipe YAML before running
- `comprexx recipe recommend` — generate a recipe from a model profile + target hardware
- `comprexx recipe diff <recipe_a.yaml> <recipe_b.yaml>` — compare two recipes
- Built-in recipe registry (ship with pre-built recipes for common model/target combinations)

**Built-in recipe registry (v1):**
- `resnet50/edge-cpu` — INT8 + structured pruning for CPU edge
- `resnet50/jetson-nano` — INT8 + filter pruning + TensorRT export
- `bert-base/mobile` — INT8 + attention head pruning
- `yolov8n/rpi4` — INT8 + filter pruning
- `mobilenetv3/wasm` — optimized for WASM/browser deployment

---

### 6.5 Accuracy Recovery

When a compression stage degrades accuracy past the guard threshold, Comprexx surfaces recovery options:

**Fine-tuning hooks:**
- Comprexx cannot run fine-tuning itself, but produces a `FineTuneConfig` describing:
  - Recommended number of fine-tuning epochs
  - Recommended learning rate schedule
  - Layers to freeze/unfreeze
- Compatible with: standard PyTorch, HuggingFace Trainer, Lightning

**QAT insertion:**
- After PTQ results in unacceptable accuracy, Comprexx can convert the PTQ model to a QAT-ready model
- User fine-tunes, returns model to Comprexx for finalization

**Distillation recovery:**
- After aggressive pruning, Comprexx can set up a distillation recipe using the original model as teacher

**Accuracy repair suggestions:**
- Automatically surface "which layers are causing the most accuracy loss" via sensitivity analysis
- Suggest excluding problematic layers from compression or relaxing their compression targets

---

### 6.6 Export & Deployment Targets

After compression, the model is exported to deployment-ready formats.

**Export formats:**

| Format | Notes |
|--------|-------|
| `pytorch` | Standard `.pt` checkpoint, includes compression metadata |
| `torchscript` | TorchScript traced or scripted model |
| `onnx` | Configurable opset (default: 17); includes shape inference |
| `onnx_quantized` | ONNX model with quantized operators |
| `tflite` | TFLite FlatBuffer via ONNX → TFLite conversion |
| `tensorrt` | TensorRT engine via `torch2trt` or ONNX→TRT pipeline |
| `coreml` | Apple CoreML via `coremltools` (optional dependency) |
| `executorch` | PyTorch ExecuTorch for mobile/embedded (optional) |

**Export metadata:**
Every exported artifact is accompanied by a `comprexx_manifest.json` containing:
- Original model name/hash
- Recipe used
- Compression stats (size reduction %, FLOPs reduction %, accuracy delta)
- Target hardware
- Export format + version
- Comprexx version
- Timestamp

**ONNX export quality:**
- Auto-run `onnxruntime` inference check post-export
- Numerical equality check (max output diff between PyTorch and ONNX output on sample inputs)
- Shape inference pass
- Simplification via `onnx-simplifier`

---

### 6.7 Visualization & Observability Dashboard

The visualization layer is what separates Comprexx from a collection of scripts.
It runs as a local web app (built with Streamlit or Panel, see tech stack discussion) launched by `comprexx visualize`.

**Dashboard views:**

#### A. Model Anatomy View
- Layer-by-layer table: name, type, shape, parameter count, FLOPs, compression status
- Treemap of parameter distribution (which blocks/layers dominate)
- Highlight compressible vs. non-compressible layers

#### B. Compression Impact View (per-stage)
- Side-by-side before/after for: model size (MB), parameter count, FLOPs, accuracy
- Waterfall chart: cumulative impact of each pipeline stage on all metrics
- Accuracy vs. Compression Ratio curve (Pareto frontier)
- Layer sensitivity heatmap (which layers were most affected by each technique)

#### C. Quantization Analysis View
- Per-layer weight distribution histograms (before/after quantization)
- Quantization error (MSE) per layer
- Outlier detection visualization (for INT4/INT8 overflow risks)
- Calibration data distribution

#### D. Pruning Analysis View
- Sparsity heatmap per layer (% of zeroed weights)
- Magnitude distribution before/after pruning per layer
- Filter/head importance scores visualization
- Structured pruning: channel-level activation statistics

#### E. Accuracy Tracker
- Evaluation metric over compression stages (top-1, top-5, F1, custom)
- Confidence interval (if multiple eval runs)
- Regression from baseline clearly marked in red if over threshold

#### F. Export Summary View
- Comparison table: original vs. each exported format
  - Model size, estimated latency (if benchmarked), precision
- Hardware compatibility matrix
- Download links for all artifacts

#### G. Recipe Builder UI (Good-to-Have, v2)
- Visual drag-and-drop pipeline builder
- Real-time estimate of expected compression based on selected techniques
- Export recipe as YAML

---

### 6.8 CLI Interface

The CLI is a primary user interface, not an afterthought.

```
comprexx <command> [options]
```

**Commands:**

| Command | Description |
|---------|-------------|
| `comprexx analyze <model>` | Profile a model and print stats |
| `comprexx compress <model> --recipe <recipe.yaml>` | Run compression pipeline |
| `comprexx compress <model> --auto --target <hw>` | Auto-recommend and run |
| `comprexx export <model.pt> --format onnx,trt` | Export a compressed model |
| `comprexx benchmark <model.onnx> --device cpu` | Benchmark inference latency |
| `comprexx visualize <run_dir>` | Launch visualization dashboard |
| `comprexx recipe validate <recipe.yaml>` | Validate a recipe file |
| `comprexx recipe recommend <model> --target <hw>` | Generate a recipe |
| `comprexx recipe diff <a.yaml> <b.yaml>` | Diff two recipes |
| `comprexx compare <run_dir_a> <run_dir_b>` | Compare two compression runs |
| `comprexx init` | Initialize a new Comprexx project in cwd |

**CLI output philosophy:**
- Progress bars for long-running operations (via `rich`)
- Structured, colored terminal output
- JSON output mode (`--json`) for scripting
- Verbose mode (`--verbose`) for debugging
- All outputs saved to a run directory (`./comprexx_runs/<run_id>/`)

---

### 6.9 Python SDK API

The SDK is the primary programmatic interface. Designed for Jupyter notebooks, scripts, and integration into training pipelines.

```python
import comprexx as cx

# 1. Load and profile
model = cx.load("torchvision.models.resnet50", weights="IMAGENET1K_V1")
profile = cx.analyze(model, input_shape=(1, 3, 224, 224))
print(profile.summary())

# 2. Build a pipeline
pipeline = cx.Pipeline([
    cx.stages.StructuredPruning(sparsity=0.30, criteria="l1_norm"),
    cx.stages.PTQStatic(format="int8", calibration_samples=512),
    cx.stages.OperatorFusion(),
])

# 3. Run with eval data for accuracy tracking
result = pipeline.run(
    model,
    calibration_data=cal_loader,
    eval_fn=my_eval_fn,
    accuracy_guard=cx.AccuracyGuard(metric="top1", max_drop=0.01),
)

# 4. Inspect results
print(result.report)
result.plot()  # opens visualization

# 5. Export
result.export("./artifacts/", formats=["onnx", "tensorrt"])
```

**Key classes:**

- `cx.Pipeline` — ordered list of stages, run configuration
- `cx.stages.*` — all compression technique classes
- `cx.analyze(model)` → `ModelProfile` — model profiling
- `cx.AccuracyGuard` — accuracy threshold configuration
- `cx.CompressionResult` — result of a pipeline run
- `cx.CompressionReport` — detailed per-stage metrics
- `cx.Recipe` — load/save/validate recipes
- `cx.Exporter` — export to various formats
- `cx.Benchmark` — latency/throughput benchmarking

**Design requirements:**
- Full type hints throughout
- Dataclass-based configuration objects (not raw dicts)
- `__repr__` on all result objects produces readable summaries
- All objects serializable to JSON
- Compatible with Python 3.10, 3.11, 3.12

---

### 6.10 CI/CD & Automation Integration

Comprexx is a first-class CI/CD citizen.

**GitHub Actions integration:**
- Official `comprexx-action` (GitHub Action wrapper)
- Compress + export on every model push
- Post compression report as PR comment (size delta, accuracy delta)
- Fail CI if accuracy drops past threshold

**Pre-built CI configs:**
- `comprexx init --ci github` generates a `.github/workflows/compress.yml`
- `comprexx init --ci gitlab` for GitLab CI

**Artifact tracking:**
- `comprexx_manifest.json` emitted with every compression run
- Compatible with MLflow, Weights & Biases (W&B), and DVC for experiment tracking
- Optional: log compression metrics directly to W&B/MLflow via `--log-to wandb`

**JSON/structured output for scripting:**
- All CLI commands support `--json` flag
- Exit codes are meaningful (0: success, 1: accuracy guard triggered, 2: export failed, etc.)

---

### 6.11 Good-to-Haves (v2+)

These features are explicitly scoped to v2 and beyond but inform v1 architecture decisions.

**v2 Features:**
- **Recipe Builder UI** — drag-and-drop web UI for building recipes visually
- **Sensitivity Profiler** — run automated sensitivity analysis to guide recipe creation
- **Hardware Profiler** — plug in a target device and benchmark actual latency (not estimated)
- **Compression-Aware NAS hooks** — feed compression constraints into architecture search
- **LLM-specific pipeline** — dedicated support for transformer decoder models (LLaMA, Mistral, Phi)
  including GPTQ, AWQ, SmoothQuant, KV cache quantization
- **WASM/Web export** — `onnxruntime-web` compatible export with size-optimized ONNX

**v3 Features:**
- **Comprexx Hub** — community recipe registry (share/discover compression recipes)
- **Managed compression service** — cloud API for compression without local GPU
- **Hardware-in-the-loop benchmarking** — submit model, get real latency from physical device fleet

---

## 7. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Comprexx Core                            │
│                                                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────┐  │
│  │   Ingestion │   │   Pipeline  │   │   Export Engine      │  │
│  │   Layer     │   │   Engine    │   │                      │  │
│  │             │   │             │   │  PyTorch / ONNX /    │  │
│  │  ModelLoader│   │  Stage      │   │  TorchScript /       │  │
│  │  Profiler   │──▶│  Orchestrat │──▶│  TensorRT /          │  │
│  │  GraphTrace │   │  or         │   │  TFLite / CoreML     │  │
│  └─────────────┘   │             │   └──────────────────────┘  │
│                    │  AccuracyGu │                              │
│  ┌─────────────┐   │  ard        │   ┌──────────────────────┐  │
│  │   Recipe    │   │             │   │   Reporting &        │  │
│  │   System    │──▶│  StageReport│──▶│   Visualization      │  │
│  │             │   │  Aggregator │   │                      │  │
│  │  Loader     │   └─────────────┘   │  Dashboard Server    │  │
│  │  Validator  │         │           │  Chart Engine        │  │
│  │  Recommender│         ▼           │  Report Serializer   │  │
│  └─────────────┘   ┌─────────────┐  └──────────────────────┘  │
│                    │  Compression│                              │
│                    │  Techniques │                              │
│                    │             │                              │
│                    │ Quantization│                              │
│                    │ Pruning     │                              │
│                    │ Distillation│                              │
│                    │ LRD         │                              │
│                    │ Fusion      │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
         │                                     │
         ▼                                     ▼
  ┌──────────────┐                    ┌──────────────────┐
  │   CLI        │                    │   Python SDK     │
  │  (Typer)     │                    │  (public API)    │
  └──────────────┘                    └──────────────────┘
```

---

## 8. Tech Stack

### Core Runtime
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Deep learning | PyTorch 2.x | Industry standard; torch.fx, quantization, TorchScript all native |
| Graph tracing | `torch.fx` | First-class graph IR; enables custom passes for fusion, LRD |
| ONNX export | `torch.onnx` + `onnxruntime` | Gold standard for cross-framework export |
| ONNX optimization | `onnxsim` (onnx-simplifier) | Remove redundant ops post-export |
| TensorRT | `tensorrt` Python API / `torch2trt` | NVIDIA GPU deployment |
| Quantization backends | `bitsandbytes`, `auto-gptq`, `torch.quantization` | Broad coverage |
| Numerical computing | NumPy | Array ops throughout |

### CLI
| Component | Choice | Rationale |
|-----------|--------|-----------|
| CLI framework | `Typer` | Type-annotated, auto-help generation, Pythonic |
| Terminal UI | `Rich` | Beautiful progress bars, tables, colored output |

### Visualization Dashboard
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Dashboard framework | `Streamlit` | Fastest time-to-dashboard; ML-native audience familiar with it |
| Charts | `Plotly` via `plotly.express` | Interactive, embeddable, consistent |
| Heatmaps / fine-grained | `matplotlib` (embedded in Streamlit) | For custom layer-level plots |

> **Note on original tech stack:** The original proposal mentioned TensorRT (kept), ONNX (kept), PyTorch (kept), NumPy (kept), and `torch.fx` (kept). Streamlit replaces a proposed custom viz layer — it provides a production-quality dashboard with far less code. If a fully custom embedded dashboard is desired in v2, React + Plotly is the right path.

### Serialization & Config
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Recipe format | YAML + Python dataclasses | Human-writable + type-safe programmatic access |
| Config validation | `pydantic` v2 | Schema validation, error messages, serialization |
| Experiment logging | Optional: `mlflow`, `wandb` | Log compression metrics alongside training metrics |

### Testing
| Component | Choice | Rationale |
|-----------|--------|-----------|
| Test runner | `pytest` | Standard |
| Coverage | `pytest-cov` | Coverage reporting |
| Model fixtures | TorchVision tiny models (ResNet-18, MobileNetV2) | Fast test cycles |
| Benchmarking | `pytest-benchmark` | Track performance regressions |

### Packaging & Distribution
| Component | Choice |
|-----------|--------|
| Package manager | `uv` (build) + pip installable |
| Distribution | PyPI (`comprexx`) |
| Python versions | 3.10, 3.11, 3.12 |
| Optional deps | `comprexx[tensorrt]`, `comprexx[tflite]`, `comprexx[coreml]` |

---

## 9. The Recipe DSL (Detailed)

### Full Recipe Schema (Pydantic)

```python
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union
from enum import Enum

class TargetHardware(str, Enum):
    CPU_GENERIC = "cpu"
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN = "jetson_orin"
    RASPBERRY_PI_4 = "rpi4"
    NVIDIA_T4 = "t4"
    NVIDIA_A100 = "a100"
    APPLE_SILICON = "apple_silicon"
    MOBILE_ANDROID = "android"
    MOBILE_IOS = "ios"
    WASM = "wasm"

class AccuracyGuard(BaseModel):
    metric: str = "top1_accuracy"  # or custom fn name
    max_drop: float = 0.02         # fractional (0.02 = 2%)
    action: Literal["halt", "warn"] = "halt"

class PruningStageConfig(BaseModel):
    technique: Literal["structured_pruning", "unstructured_pruning", "nm_sparsity"]
    sparsity: float = Field(ge=0.0, le=1.0)
    criteria: Literal["l1_norm", "l2_norm", "taylor", "activation_stats", "random"] = "l1_norm"
    scope: Literal["global", "local"] = "global"
    target: Literal["filters", "channels", "heads", "layers", "weights"] = "filters"
    schedule: Literal["oneshot", "iterative"] = "oneshot"
    exclude_layers: list[str] = []

class QuantizationStageConfig(BaseModel):
    technique: Literal["ptq_dynamic", "ptq_static", "ptq_weight_only", "qat"]
    format: Literal["int8", "int4", "nf4", "fp8", "fp16"] = "int8"
    algorithm: Literal["rtn", "gptq", "awq", "smoothquant"] = "rtn"
    calibration_method: Literal["minmax", "percentile", "entropy", "omse"] = "percentile"
    calibration_samples: int = 512
    exclude_layers: list[str] = []

class LRDStageConfig(BaseModel):
    technique: Literal["low_rank_decomposition"]
    rank_ratio: Optional[float] = None  # fraction of original rank
    energy_threshold: Optional[float] = 0.90  # retain 90% of spectral energy
    target_layers: list[str] = []  # empty = all applicable

class FusionStageConfig(BaseModel):
    technique: Literal["operator_fusion"]
    patterns: list[str] = ["conv_bn_relu", "linear_gelu", "linear_layernorm"]

class ExportConfig(BaseModel):
    formats: list[Literal["pytorch", "torchscript", "onnx", "onnx_quantized",
                           "tensorrt", "tflite", "coreml", "executorch"]]
    onnx_opset: int = 17
    tensorrt_precision: Literal["fp32", "fp16", "int8"] = "fp16"
    output_dir: str = "./comprexx_artifacts"

StageConfig = Union[PruningStageConfig, QuantizationStageConfig,
                    LRDStageConfig, FusionStageConfig]

class Recipe(BaseModel):
    name: str
    version: str = "1.0"
    description: str = ""
    target_hardware: Optional[TargetHardware] = None
    base_model: str  # module path, HF model id, or file path
    accuracy_guard: Optional[AccuracyGuard] = None
    stages: list[StageConfig]
    export: Optional[ExportConfig] = None
```

---

## 10. Visualization Specs

### Dashboard: Component Specifications

#### Model Anatomy View
- Input: `ModelProfile` JSON
- Components:
  - Summary card (total params, FLOPs, size in MB)
  - Treemap: layers as rectangles, area ∝ parameter count, color = layer type
  - Table: sortable by parameter count, FLOPs; filterable by layer type
  - "Compressibility" score column (0–10 heuristic score, colored green → red)

#### Compression Impact Waterfall
- Input: `CompressionReport`
- X-axis: pipeline stages (baseline → stage 1 → stage 2 → ...)
- Y-axes (multi-panel):
  - Model size (MB)
  - FLOPs (×10^9)
  - Accuracy (%)
- Display type: waterfall chart (delta at each stage shown as positive/negative bar)
- Threshold line shown on accuracy panel

#### Accuracy vs. Compression Pareto Curve
- Input: multiple `CompressionReport` from a parameter sweep
- X-axis: compression ratio (original size / compressed size)
- Y-axis: accuracy (%)
- Each point = one configuration
- Pareto frontier highlighted
- Hover tooltip: show full recipe config for that point

#### Layer Sensitivity Heatmap
- Input: sensitivity analysis output
- X-axis: compression technique
- Y-axis: layer name
- Color: accuracy drop when that layer is compressed at a given level
- Helps identify which layers to protect (`exclude_layers`)

#### Quantization Distribution Plots
- Per-layer weight histograms with overlay of quantization grid
- Shown before and after quantization side by side
- Highlight outlier weights that cause clipping

---

## 11. CLI Specification

### `comprexx analyze`

```
comprexx analyze <model_source> [OPTIONS]

Arguments:
  model_source        Module path (e.g., torchvision.models.resnet50),
                      local .pt file, or HuggingFace model ID

Options:
  --input-shape       Input tensor shape, e.g. "1,3,224,224" [required]
  --device            cpu|cuda [default: cpu]
  --output            Path to save profile JSON
  --json              Output JSON to stdout
  --verbose           Print layer-by-layer breakdown
```

**Example output:**
```
╔══════════════════════════════════════════════╗
║  Comprexx Model Analysis: resnet50           ║
╠══════════════════════════════════════════════╣
║  Parameters:     25,557,032 (25.6M)          ║
║  FLOPs:          4.09 GFLOPs                 ║
║  Model size:     97.5 MB (FP32)              ║
║  Architecture:   CNN (ResNet-style)          ║
╠══════════════════════════════════════════════╣
║  Top compressible layers:                    ║
║   layer3.*   → 47% of FLOPs  (prune target)  ║
║   layer4.*   → 31% of FLOPs  (prune target)  ║
║   layer1.*   → 8%  of FLOPs  (low priority)  ║
╠══════════════════════════════════════════════╣
║  Suggested recipe: resnet50/edge-cpu         ║
║  Run: comprexx recipe recommend resnet50     ║
╚══════════════════════════════════════════════╝
```

### `comprexx compress`

```
comprexx compress <model_source> [OPTIONS]

Options:
  --recipe            Path to recipe YAML file
  --auto              Auto-generate recipe based on --target
  --target            Target hardware (see TargetHardware enum)
  --eval-fn           Python dotted path to evaluation function
  --calibration-data  Path to calibration dataset (for PTQ static)
  --dry-run           Estimate only, no compression applied
  --output-dir        Override recipe output dir
  --interactive       Pause between stages for inspection
  --json              Output report as JSON
```

### `comprexx benchmark`

```
comprexx benchmark <model_path> [OPTIONS]

Options:
  --device            cpu|cuda|mps
  --backend           pytorch|onnxruntime|tensorrt
  --batch-size        Batch size for benchmarking [default: 1]
  --warmup-runs       Number of warmup runs [default: 20]
  --benchmark-runs    Number of benchmark runs [default: 100]
  --input-shape       Input tensor shape
```

---

## 12. Python SDK Specification

### Core Module Structure

```
comprexx/
├── __init__.py          # Public API surface
├── core/
│   ├── pipeline.py      # Pipeline, PipelineResult
│   ├── report.py        # CompressionReport, StageReport
│   ├── guard.py         # AccuracyGuard
│   └── exceptions.py    # ComprexxError hierarchy
├── analysis/
│   ├── profiler.py      # ModelProfile, analyze()
│   ├── sensitivity.py   # SensitivityAnalyzer
│   └── flops.py         # FLOPs / MACs counter
├── stages/
│   ├── base.py          # CompressionStage ABC
│   ├── quantization/
│   │   ├── ptq_dynamic.py
│   │   ├── ptq_static.py
│   │   ├── ptq_weight_only.py
│   │   └── qat.py
│   ├── pruning/
│   │   ├── structured.py
│   │   ├── unstructured.py
│   │   └── nm_sparsity.py
│   ├── distillation/
│   │   └── trainer.py
│   ├── lrd.py           # Low-rank decomposition
│   └── fusion.py        # Operator fusion
├── recipe/
│   ├── schema.py        # Pydantic schemas
│   ├── loader.py        # Load/save/validate
│   └── recommender.py   # Auto-recipe generation
├── export/
│   ├── onnx.py
│   ├── torchscript.py
│   ├── tensorrt.py
│   ├── tflite.py
│   └── manifest.py      # Artifact manifest
├── benchmark/
│   └── latency.py       # Latency/throughput benchmarking
├── viz/
│   ├── dashboard.py     # Streamlit dashboard entry point
│   └── charts.py        # Plotly chart generators
└── cli/
    └── main.py          # Typer CLI entry point
```

### Public API (`comprexx/__init__.py`)

```python
# Analysis
from comprexx import analyze, ModelProfile

# Pipeline
from comprexx import Pipeline
from comprexx import stages
from comprexx.core import AccuracyGuard

# Recipe
from comprexx import Recipe, load_recipe

# Export
from comprexx import export

# Benchmark
from comprexx import benchmark

# Loader
from comprexx import load_model
```

### `CompressionStage` ABC

```python
class CompressionStage(ABC):
    name: str
    config: BaseModel

    @abstractmethod
    def apply(self, model: nn.Module, context: StageContext) -> tuple[nn.Module, StageReport]:
        """Apply compression. Return (compressed_model, report)."""

    def estimate(self, profile: ModelProfile) -> StageEstimate:
        """Estimate compression impact without applying. May be approximate."""
```

### `StageReport`

```python
@dataclass
class StageReport:
    stage_name: str
    technique: str
    duration_seconds: float
    param_count_before: int
    param_count_after: int
    flops_before: int
    flops_after: int
    size_bytes_before: int
    size_bytes_after: int
    accuracy_before: Optional[float]
    accuracy_after: Optional[float]
    accuracy_delta: Optional[float]
    notes: list[str]

    @property
    def compression_ratio(self) -> float: ...
    @property
    def flops_reduction_pct(self) -> float: ...
    @property
    def size_reduction_pct(self) -> float: ...
```

---

## 13. Export Targets Specification

### PyTorch Export
- Save compressed `nn.Module` as `.pt` checkpoint
- Include `compression_metadata` in state dict
- Validate: reload and run forward pass

### ONNX Export
- Use `torch.onnx.export` with dynamic axes support
- Run `onnxsim` simplification pass
- Run `onnxruntime` forward pass for numerical validation
- Max output difference threshold: `1e-5` (configurable)
- Include metadata in ONNX model properties

### TensorRT Export
- Path: PyTorch → ONNX → TensorRT engine
- Support FP32, FP16, INT8 (INT8 requires calibration data)
- Engine serialized to `.trt` file
- Profile multiple batch sizes (1, 4, 8, 16, 32 by default)

### TFLite Export
- Path: PyTorch → ONNX → TFLite (via `onnx-tf` + `tf.lite.TFLiteConverter`)
- Post-training quantization during TFLite conversion (optional)
- `.tflite` FlatBuffer output
- Validation via `tflite-runtime`

### TorchScript Export
- Support both `torch.jit.trace` and `torch.jit.script` modes
- Trace mode: requires sample input
- Script mode: requires scriptable model (warns if tracing used as fallback)

---

## 14. Benchmark & Evaluation Framework

### Latency Benchmarking

```python
result = cx.benchmark(
    model_path="./artifacts/resnet50.onnx",
    backend="onnxruntime",
    device="cpu",
    input_shape=(1, 3, 224, 224),
    warmup_runs=20,
    benchmark_runs=100,
)
# result.mean_latency_ms, result.p95_latency_ms, result.throughput_qps
```

**Backends:** `pytorch`, `onnxruntime`, `tensorrt`
**Metrics:** mean latency (ms), P50/P95/P99 latency, throughput (queries/s), peak memory (MB)

### Accuracy Evaluation

Users provide an `eval_fn: Callable[[nn.Module], dict[str, float]]` — a function that takes a model and returns accuracy metrics as a dict.

```python
def my_eval(model):
    # run eval on your dataset
    return {"top1": 0.756, "top5": 0.924}
```

This is deliberately not abstracted — Comprexx should not own the eval loop.

### Sensitivity Analysis

```python
sensitivity = cx.analyze_sensitivity(
    model,
    technique="ptq_static",
    eval_fn=my_eval,
    granularity="layer",  # or "block"
)
# sensitivity.plot_heatmap()
# sensitivity.most_sensitive_layers(n=5)
```

Runs the specified technique individually on each layer (or block) and records accuracy impact — O(N) eval passes where N = number of layers.

---

## 15. Configuration & Persistence

### Run Directory Structure

Every `compress` or `analyze` run produces a run directory:

```
comprexx_runs/
└── run_20250405_143022_resnet50/
    ├── comprexx_run.json          # Full run metadata
    ├── model_profile.json         # Original model profile
    ├── recipe_used.yaml           # Recipe that was run
    ├── compression_report.json    # Full compression report
    ├── stage_reports/
    │   ├── 01_structured_pruning.json
    │   └── 02_ptq_static.json
    ├── artifacts/
    │   ├── model_compressed.pt
    │   ├── model_compressed.onnx
    │   └── comprexx_manifest.json
    └── viz_data/
        └── dashboard_data.json    # Pre-computed viz data for dashboard
```

### Global Config (`~/.comprexx/config.yaml`)

```yaml
default_output_dir: ~/comprexx_runs
default_device: cpu
telemetry: false  # never send usage data without explicit opt-in
log_level: INFO
onnx_opset: 17
```

---

## 16. Error Handling & Developer Experience

### Exception Hierarchy

```python
class ComprexxError(Exception): pass

class ModelLoadError(ComprexxError): pass
class RecipeValidationError(ComprexxError): pass
class AccuracyGuardTriggered(ComprexxError):
    stage: str
    baseline: float
    current: float
    threshold: float
class ExportError(ComprexxError): pass
class CalibrationError(ComprexxError): pass
class UnsupportedLayerError(ComprexxError): pass
```

### Error Messages

Errors should follow the pattern: **what happened** → **why** → **what to do**.

```
AccuracyGuardTriggered: Accuracy dropped 3.2% after stage 'ptq_static',
exceeding the configured threshold of 1.0%.

  Baseline top1:    76.1%
  After stage:      72.9%
  Configured limit: 1.0% drop max

  Suggestions:
  → Exclude sensitive layers: add to exclude_layers: ["layer4.1.conv2"]
  → Use a less aggressive calibration: change format from 'int4' to 'int8'
  → Add QAT recovery: insert a qat stage after ptq_static and fine-tune
  → Run sensitivity analysis: comprexx analyze --sensitivity resnet50
```

### Warnings System

Non-fatal issues emit structured warnings:

- `LayerSkippedWarning` — a layer was not compressible, skipped
- `CalibrationDataWarning` — calibration dataset smaller than recommended
- `PrecisionLossWarning` — quantization error on a layer exceeds a soft threshold
- `ExportNumericalWarning` — ONNX output differs from PyTorch beyond tolerance

---

## 17. Testing Strategy

### Unit Tests
- Each `CompressionStage` has isolated tests with a tiny model fixture
- Recipe validation: valid + invalid YAML inputs
- Export: each format tested with a 3-layer toy model
- Report serialization round-trips

### Integration Tests
- Full pipeline runs on ResNet-18 (fast, ~11M params)
- Full pipeline runs on a tiny Transformer (2-layer, 2-head, d_model=64)
- CLI commands tested via `subprocess` / `Typer`'s `CliRunner`

### Regression Tests
- Compression ratio must be within ±2% of expected for each technique at standard configs
- Accuracy drop must be within expected range on CIFAR-10 with ResNet-18

### Performance Tests
- Pipeline run time on ResNet-18 must complete in < 60s on CPU (no calibration)
- Memory usage during profiling must not exceed 2× model size

### Model Fixtures

```python
# tests/fixtures/models.py
def tiny_resnet():      # ResNet-18 with 2 layers
def tiny_transformer(): # 2-layer encoder, d_model=64
def tiny_cnn():         # 3-layer ConvNet
```

---

## 18. Repository Structure

```
comprexx/
├── comprexx/                   # Main package
│   ├── __init__.py
│   ├── core/
│   ├── analysis/
│   ├── stages/
│   ├── recipe/
│   ├── export/
│   ├── benchmark/
│   ├── viz/
│   └── cli/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── recipes/                    # Built-in recipe registry
│   ├── resnet50/
│   │   ├── edge-cpu.yaml
│   │   └── jetson-nano.yaml
│   ├── bert-base/
│   │   └── mobile.yaml
│   └── ...
├── docs/
│   ├── quickstart.md
│   ├── recipes.md
│   ├── techniques/
│   │   ├── quantization.md
│   │   ├── pruning.md
│   │   └── distillation.md
│   ├── api/                    # Auto-generated from docstrings
│   └── examples/
│       ├── resnet50_edge.ipynb
│       └── bert_mobile.ipynb
├── examples/
│   ├── resnet50_compress.py
│   └── transformer_compress.py
├── benchmarks/                 # Reproducible benchmark scripts
├── pyproject.toml
├── uv.lock
├── README.md
├── CHANGELOG.md
└── .github/
    ├── workflows/
    │   ├── test.yml
    │   ├── release.yml
    │   └── compress.yml        # Example compression CI
    └── ISSUE_TEMPLATE/
```

---

## 19. Roadmap & Versioning

### v0.1 — Foundation (Private Alpha)
- Model analysis + profiling
- Structured pruning (filter-level)
- PTQ static + dynamic (INT8)
- ONNX export
- CLI: `analyze`, `compress`, `export`
- Python SDK: `Pipeline`, `analyze()`, basic stages
- Minimal terminal report output

### v0.2 — Pipeline Completeness
- All quantization variants (PTQ weight-only, INT4, GPTQ)
- Unstructured pruning + N:M sparsity
- Low-rank decomposition
- Operator fusion via torch.fx
- Accuracy guard
- Full compression report
- Run directory with artifact persistence

### v0.3 — Recipe System
- Recipe YAML DSL + Pydantic schema
- `recipe validate`, `recipe recommend`, `recipe diff`
- Built-in recipe registry (5 model/target combos)

### v0.4 — Visualization
- Streamlit dashboard
- All 6 visualization views
- `comprexx visualize` command
- Export summary view

### v0.5 — Export Completeness
- TorchScript, TensorRT, TFLite export
- `comprexx_manifest.json`
- `comprexx benchmark` command

### v1.0 — Public Release
- Docs site
- Jupyter notebook examples
- GitHub Action
- W&B / MLflow integration hooks
- Stable public API
- PyPI release (`pip install comprexx`)

### v1.x — LLM Support
- Dedicated transformer decoder pipeline
- GPTQ, AWQ, SmoothQuant, KV cache quantization
- Attention head pruning improvements
- HuggingFace Hub model loading

### v2.0 — Platform
- Recipe Builder UI
- Sensitivity profiler (automated)
- Hardware-in-the-loop benchmarking (Jetson, RPi)
- Comprexx Hub (community recipe registry)

---

## 20. Competitive Positioning

| Tool | Scope | Composability | Observability | Hardware-aware | Exporters | Recipe System |
|------|-------|---------------|---------------|----------------|-----------|---------------|
| **Comprexx** | Any PyTorch model | ✅ Full pipeline | ✅ Interactive dashboard | ✅ Target-aware | ✅ PyTorch/ONNX/TRT/TFLite | ✅ Versioned YAML |
| Neural Magic LLM Compressor | LLMs only | Partial (oneshot) | ❌ None | ❌ No | ONNX/vLLM | Partial (JSON) |
| TensorRT | Inference only | ❌ No compression | ❌ No | ✅ NVIDIA only | TRT engine | ❌ |
| bitsandbytes | Quantization only | ❌ | ❌ | ❌ | ❌ | ❌ |
| torch.quantization | Quantization only | ❌ | ❌ | ❌ | Partial | ❌ |
| torch.nn.utils.prune | Pruning only | ❌ | ❌ | ❌ | ❌ | ❌ |
| Optimum (HF) | HF models only | Partial | ❌ | Partial | ONNX/ORT | ❌ |

**Comprexx's differentiator in one line:**
> The first compression toolkit where the entire pipeline — from analysis to multi-technique compression to hardware-aware export — is a single, observable, reproducible, shareable artifact.

---

*Specification version: 1.0*
*Date: April 2026*
*Project: Comprexx*
