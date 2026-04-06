"""Recipe loader — load, validate, and convert YAML recipes to pipelines."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from comprexx.core.exceptions import RecipeValidationError
from comprexx.core.guard import AccuracyGuard
from comprexx.core.pipeline import Pipeline
from comprexx.recipe.schema import RecipeV1
from comprexx.stages.base import CompressionStage
from comprexx.stages.clustering.weight_clustering import WeightClustering
from comprexx.stages.decomposition.low_rank import LowRankDecomposition
from comprexx.stages.fusion.operator_fusion import OperatorFusion
from comprexx.stages.pruning.nm_sparsity import NMSparsity
from comprexx.stages.pruning.structured import StructuredPruning
from comprexx.stages.pruning.unstructured import UnstructuredPruning
from comprexx.stages.quantization.ptq_dynamic import PTQDynamic
from comprexx.stages.quantization.ptq_static import PTQStatic
from comprexx.stages.quantization.weight_only import WeightOnlyQuant


def load_recipe(path: str | Path) -> RecipeV1:
    """Load and validate a YAML recipe file.

    Args:
        path: Path to the recipe YAML file.

    Returns:
        Validated RecipeV1 object.

    Raises:
        RecipeValidationError: If the YAML is invalid or doesn't match schema.
    """
    path = Path(path)
    if not path.exists():
        raise RecipeValidationError(f"Recipe file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise RecipeValidationError(f"Invalid YAML: {e}") from e

    if not isinstance(raw, dict):
        raise RecipeValidationError("Recipe must be a YAML mapping (dict).")

    try:
        return RecipeV1(**raw)
    except ValidationError as e:
        raise RecipeValidationError(f"Recipe validation failed:\n{e}") from e


def recipe_to_pipeline(recipe: RecipeV1) -> tuple[Pipeline, AccuracyGuard | None]:
    """Convert a validated recipe into a Pipeline with instantiated stages.

    Returns:
        Tuple of (Pipeline, optional AccuracyGuard).
    """
    stages: list[CompressionStage] = []

    for stage_config in recipe.stages:
        technique = stage_config.technique

        if technique == "structured_pruning":
            stages.append(
                StructuredPruning(
                    sparsity=stage_config.sparsity,
                    criteria=stage_config.criteria,
                    scope=stage_config.scope,
                    target=stage_config.target,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "unstructured_pruning":
            stages.append(
                UnstructuredPruning(
                    sparsity=stage_config.sparsity,
                    criteria=stage_config.criteria,
                    scope=stage_config.scope,
                    gradual_steps=stage_config.gradual_steps,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "nm_sparsity":
            stages.append(
                NMSparsity(
                    n=stage_config.n,
                    m=stage_config.m,
                    criteria=stage_config.criteria,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "ptq_dynamic":
            stages.append(PTQDynamic(format=stage_config.format))
        elif technique == "ptq_static":
            stages.append(
                PTQStatic(
                    format=stage_config.format,
                    calibration_method=stage_config.calibration_method,
                    calibration_samples=stage_config.calibration_samples,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "weight_only_quant":
            stages.append(
                WeightOnlyQuant(
                    bits=stage_config.bits,
                    group_size=stage_config.group_size,
                    symmetric=stage_config.symmetric,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "low_rank_decomposition":
            stages.append(
                LowRankDecomposition(
                    rank_ratio=stage_config.rank_ratio,
                    energy_threshold=stage_config.energy_threshold,
                    mode=stage_config.mode,
                    min_rank=stage_config.min_rank,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        elif technique == "operator_fusion":
            stages.append(
                OperatorFusion(
                    fuse_conv_bn=stage_config.fuse_conv_bn,
                    fallback_on_trace_error=stage_config.fallback_on_trace_error,
                )
            )
        elif technique == "weight_clustering":
            stages.append(
                WeightClustering(
                    num_clusters=stage_config.num_clusters,
                    init=stage_config.init,
                    max_iter=stage_config.max_iter,
                    per_layer=stage_config.per_layer,
                    exclude_layers=stage_config.exclude_layers,
                )
            )
        else:
            raise RecipeValidationError(f"Unknown technique: {technique}")

    guard = None
    if recipe.accuracy_guard is not None:
        guard = AccuracyGuard(
            metric=recipe.accuracy_guard.metric,
            max_drop=recipe.accuracy_guard.max_drop,
            action=recipe.accuracy_guard.action,
        )

    return Pipeline(stages), guard
