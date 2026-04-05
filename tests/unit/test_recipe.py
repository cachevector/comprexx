"""Tests for recipe schema and loader."""

import pytest

from comprexx.core.exceptions import RecipeValidationError
from comprexx.recipe.loader import load_recipe, recipe_to_pipeline
from comprexx.recipe.schema import RecipeV1


VALID_RECIPE_YAML = """\
name: test-recipe
version: "1.0"
description: "Test recipe"
base_model: test_model

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
"""

INVALID_RECIPE_YAML = """\
name: bad-recipe
stages:
  - technique: unknown_technique
    sparsity: 0.5
"""


class TestRecipeSchema:
    def test_valid_recipe(self):
        recipe = RecipeV1(
            name="test",
            stages=[
                {"technique": "structured_pruning", "sparsity": 0.3},
                {"technique": "ptq_dynamic"},
            ],
        )
        assert recipe.name == "test"
        assert len(recipe.stages) == 2

    def test_discriminated_union(self):
        recipe = RecipeV1(
            name="test",
            stages=[{"technique": "ptq_static", "calibration_method": "entropy"}],
        )
        assert recipe.stages[0].technique == "ptq_static"
        assert recipe.stages[0].calibration_method == "entropy"


class TestLoadRecipe:
    def test_load_valid(self, tmp_path):
        p = tmp_path / "recipe.yaml"
        p.write_text(VALID_RECIPE_YAML)
        recipe = load_recipe(p)
        assert recipe.name == "test-recipe"
        assert len(recipe.stages) == 2
        assert recipe.accuracy_guard is not None
        assert recipe.accuracy_guard.max_drop == 0.02

    def test_load_missing_file(self):
        with pytest.raises(RecipeValidationError, match="not found"):
            load_recipe("/nonexistent/recipe.yaml")

    def test_load_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("{{invalid yaml")
        with pytest.raises(RecipeValidationError):
            load_recipe(p)

    def test_load_invalid_schema(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text(INVALID_RECIPE_YAML)
        with pytest.raises(RecipeValidationError, match="validation failed"):
            load_recipe(p)


class TestRecipeToPipeline:
    def test_converts_to_pipeline(self, tmp_path):
        p = tmp_path / "recipe.yaml"
        p.write_text(VALID_RECIPE_YAML)
        recipe = load_recipe(p)
        pipeline, guard = recipe_to_pipeline(recipe)

        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "structured_pruning"
        assert pipeline.stages[1].name == "ptq_dynamic"
        assert guard is not None
        assert guard.max_drop == 0.02

    def test_no_guard(self):
        recipe = RecipeV1(
            name="test",
            stages=[{"technique": "ptq_dynamic"}],
        )
        pipeline, guard = recipe_to_pipeline(recipe)
        assert guard is None
