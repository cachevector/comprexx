<p align="center">
  <img src="./docs/assets/comprexx-logo.svg" alt="Comprexx Logo" width="100" />
</p>

<h1 align="center">Comprexx</h1>

<p align="center">
  <b>Compress smarter. Ship faster. Run anywhere.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/License-Apache_2.0-green" />
  <img src="https://img.shields.io/badge/python-3.10+-blue" />
  <img src="https://img.shields.io/badge/version-0.1.0-orange" />
</p>

---

Comprexx is an open-source model compression toolkit for PyTorch. It takes your trained model and runs it through a pipeline of compression techniques (pruning, quantization, etc.), then exports it to a deployment-ready format. At every step, it tells you exactly what changed: how much smaller the model got, how many FLOPs were saved, and what it cost in accuracy.

No more gluing together five different libraries to get a model out the door.

## Install

```bash
pip install -e ".[dev,onnx]"
```

Requires Python 3.10+ and PyTorch 2.0+.

## Usage

### Analyze a model

Before compressing anything, see what you're working with:

```python
import torch.nn as nn
import comprexx as cx

model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 10),
)

profile = cx.analyze(model, input_shape=(1, 3, 224, 224))
print(profile.summary())
```

This gives you total parameter count, FLOPs, model size, architecture type, and a per-layer breakdown showing which layers are worth compressing.

### Compress a model

Build a pipeline of compression stages and run it:

```python
pipeline = cx.Pipeline([
    cx.stages.StructuredPruning(sparsity=0.3, criteria="l1_norm"),
    cx.stages.PTQDynamic(),
])

result = pipeline.run(model, input_shape=(1, 3, 224, 224))
print(result.report.summary())
```

`StructuredPruning` ranks conv filters by importance and zeros out the bottom 30%. `PTQDynamic` quantizes Linear layers to INT8 at runtime. You can chain as many stages as you want.

The result gives you the compressed model and a report with before/after metrics for each stage.

### Set accuracy guards

If you have an eval function, the pipeline can halt automatically when accuracy drops too far:

```python
def eval_fn(model):
    # run your evaluation
    return {"top1_accuracy": 0.92}

result = pipeline.run(
    model,
    input_shape=(1, 3, 224, 224),
    eval_fn=eval_fn,
    accuracy_guard=cx.AccuracyGuard(metric="top1_accuracy", max_drop=0.02),
)
```

If accuracy drops more than 2%, the pipeline stops and tells you which stage caused the problem and what to try instead.

### Export to ONNX

```python
exporter = cx.ONNXExporter()
exporter.export(result.model, input_shape=(1, 3, 224, 224), output_path="model.onnx")
```

This runs `torch.onnx.export`, optionally simplifies the graph with `onnxsim`, and validates the output against the PyTorch model. A `comprexx_manifest.json` is saved alongside the model with compression stats and metadata.

### Use recipes

Instead of writing Python, define your pipeline as YAML:

```yaml
name: resnet-edge
description: "Pruned and quantized for edge deployment"

accuracy_guard:
  metric: top1_accuracy
  max_drop: 0.02
  action: halt

stages:
  - technique: structured_pruning
    sparsity: 0.3
    criteria: l1_norm
    scope: global

  - technique: ptq_dynamic
    format: int8
```

Load and run it:

```python
recipe = cx.load_recipe("resnet-edge.yaml")

from comprexx.recipe.loader import recipe_to_pipeline
pipeline, guard = recipe_to_pipeline(recipe)
result = pipeline.run(model, input_shape=(1, 3, 224, 224), accuracy_guard=guard)
```

### CLI

Everything above is also available from the command line:

```bash
# Analyze
comprexx analyze model.pt --input-shape "1,3,224,224"
comprexx analyze model.pt --input-shape "1,3,224,224" --verbose
comprexx analyze model.pt --input-shape "1,3,224,224" --json

# Compress with a recipe
comprexx compress model.pt --recipe recipe.yaml --input-shape "1,3,224,224"
comprexx compress model.pt --recipe recipe.yaml --input-shape "1,3,224,224" --dry-run

# Export
comprexx export model.pt --format onnx --input-shape "1,3,224,224"
```

Every compression run saves its artifacts (model profile, compression report, per-stage reports) to a `comprexx_runs/` directory so you can compare runs later.

## Available techniques

| Technique | Description |
|-----------|-------------|
| Structured pruning | Removes entire conv filters ranked by L1/L2 norm. Supports global and per-layer scoping, with `exclude_layers` to protect sensitive layers. |
| PTQ Dynamic (INT8) | Quantizes Linear and LSTM weights to INT8 at runtime. No calibration data needed. |
| PTQ Static (INT8) | Quantizes weights and activations to INT8 using calibration data to determine ranges. |

## License

Apache 2.0
