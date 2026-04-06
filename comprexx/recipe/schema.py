"""Recipe schema — Pydantic models for compression recipes."""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field


class AccuracyGuardConfig(BaseModel):
    """Accuracy guard configuration within a recipe."""

    metric: str = "top1_accuracy"
    max_drop: float = 0.02
    action: Literal["halt", "warn"] = "halt"


class StructuredPruningStageConfig(BaseModel):
    """Recipe stage config for structured pruning."""

    technique: Literal["structured_pruning"]
    sparsity: float = Field(default=0.3, ge=0.0, le=1.0)
    criteria: Literal["l1_norm", "l2_norm", "random"] = "l1_norm"
    scope: Literal["global", "local"] = "global"
    target: Literal["filters", "channels"] = "filters"
    exclude_layers: list[str] = Field(default_factory=list)


class UnstructuredPruningStageConfig(BaseModel):
    """Recipe stage config for unstructured pruning."""

    technique: Literal["unstructured_pruning"]
    sparsity: float = Field(default=0.5, ge=0.0, le=1.0)
    criteria: Literal["magnitude", "random"] = "magnitude"
    scope: Literal["global", "local"] = "global"
    gradual_steps: int = Field(default=1, ge=1)
    exclude_layers: list[str] = Field(default_factory=list)


class NMSparsityStageConfig(BaseModel):
    """Recipe stage config for N:M sparsity."""

    technique: Literal["nm_sparsity"]
    n: int = Field(default=2, ge=1)
    m: int = Field(default=4, ge=2)
    criteria: Literal["magnitude", "random"] = "magnitude"
    exclude_layers: list[str] = Field(default_factory=list)


class PTQDynamicStageConfig(BaseModel):
    """Recipe stage config for dynamic quantization."""

    technique: Literal["ptq_dynamic"]
    format: Literal["int8"] = "int8"


class PTQStaticStageConfig(BaseModel):
    """Recipe stage config for static quantization."""

    technique: Literal["ptq_static"]
    format: Literal["int8"] = "int8"
    calibration_method: Literal["minmax", "percentile", "entropy"] = "minmax"
    calibration_samples: int = 512
    exclude_layers: list[str] = Field(default_factory=list)


class WeightOnlyQuantStageConfig(BaseModel):
    """Recipe stage config for weight-only quantization."""

    technique: Literal["weight_only_quant"]
    bits: Literal[4, 8] = 8
    group_size: int = Field(default=128, ge=8)
    symmetric: bool = True
    exclude_layers: list[str] = Field(default_factory=list)


StageConfig = Annotated[
    Union[
        StructuredPruningStageConfig,
        UnstructuredPruningStageConfig,
        NMSparsityStageConfig,
        PTQDynamicStageConfig,
        PTQStaticStageConfig,
        WeightOnlyQuantStageConfig,
    ],
    Field(discriminator="technique"),
]


class ExportConfig(BaseModel):
    """Export configuration within a recipe."""

    formats: list[str] = Field(default_factory=lambda: ["pytorch"])
    onnx_opset: int = 17
    output_dir: str = "./comprexx_artifacts"


class RecipeV1(BaseModel):
    """A compression recipe — fully describes a compression pipeline."""

    name: str
    version: str = "1.0"
    description: str = ""
    base_model: str = ""
    accuracy_guard: Optional[AccuracyGuardConfig] = None
    stages: list[StageConfig]
    export: Optional[ExportConfig] = None
