"""Comprexx CLI — primary command-line interface."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="comprexx",
    help="Comprexx — ML Model Compression Toolkit",
    no_args_is_help=True,
)
console = Console()


def _parse_input_shape(shape_str: str) -> tuple[int, ...]:
    """Parse '1,3,224,224' into (1, 3, 224, 224)."""
    try:
        return tuple(int(x.strip()) for x in shape_str.split(","))
    except ValueError:
        console.print(f"[red]Invalid input shape: {shape_str}[/red]")
        raise typer.Exit(1)


def _load_model(model_source: str):
    """Load a model from a module path, .pt file, or torchvision path."""
    import torch
    import torch.nn as nn

    source = Path(model_source)

    # Local .pt file
    if source.suffix in (".pt", ".pth") and source.exists():
        try:
            model = torch.load(str(source), map_location="cpu", weights_only=False)
            if isinstance(model, nn.Module):
                return model
            if isinstance(model, dict) and "model" in model:
                return model["model"]
            console.print("[red]File does not contain a valid nn.Module[/red]")
            raise typer.Exit(1)
        except Exception as e:
            console.print(f"[red]Failed to load model from {source}: {e}[/red]")
            raise typer.Exit(1)

    # Module path like torchvision.models.resnet18
    try:
        parts = model_source.rsplit(".", 1)
        if len(parts) == 2:
            module = importlib.import_module(parts[0])
            model_fn = getattr(module, parts[1])
            return model_fn()
        else:
            module = importlib.import_module(model_source)
            return module
    except (ImportError, AttributeError) as e:
        console.print(f"[red]Cannot load model '{model_source}': {e}[/red]")
        raise typer.Exit(1)


@app.command()
def analyze(
    model_source: str = typer.Argument(..., help="Model path or Python module path"),
    input_shape: str = typer.Option(..., "--input-shape", help="Input shape, e.g. '1,3,224,224'"),
    device: str = typer.Option("cpu", help="Device (cpu or cuda)"),
    output: Optional[str] = typer.Option(None, help="Save profile JSON to path"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    verbose: bool = typer.Option(False, "--verbose", help="Show layer-by-layer breakdown"),
):
    """Profile a model and print stats."""
    from comprexx.analysis.profiler import analyze as do_analyze

    shape = _parse_input_shape(input_shape)
    model = _load_model(model_source)
    profile = do_analyze(model, shape, device, model_name=model_source.split(".")[-1])

    if json_output:
        console.print(profile.to_json())
    else:
        # Rich-formatted output
        panel_text = (
            f"[bold]Parameters:[/bold]     {profile.total_params:,} "
            f"({profile.total_params / 1e6:.1f}M)\n"
            f"[bold]FLOPs:[/bold]          {profile.total_flops / 1e9:.2f} GFLOPs\n"
            f"[bold]Model size:[/bold]     {profile.size_mb:.2f} MB\n"
            f"[bold]Architecture:[/bold]   {profile.architecture_category}\n"
            f"[bold]Layers:[/bold]         {len(profile.layers)} "
            f"({len(profile.compressible_layers())} compressible)"
        )
        console.print(Panel(panel_text, title=f"Comprexx Model Analysis: {profile.model_name}"))

        if verbose:
            table = Table(title="Layer Breakdown")
            table.add_column("Layer", style="cyan")
            table.add_column("Type")
            table.add_column("Params", justify="right")
            table.add_column("FLOPs", justify="right")
            table.add_column("Compressible", justify="center")

            for layer in profile.layers:
                table.add_row(
                    layer.name,
                    layer.layer_type,
                    f"{layer.param_count:,}",
                    f"{layer.flops:,}",
                    "[green]Yes[/green]" if layer.is_compressible else "[dim]No[/dim]",
                )
            console.print(table)

    if output:
        profile.save(output)
        console.print(f"Profile saved to {output}")


@app.command()
def compress(
    model_source: str = typer.Argument(..., help="Model path or Python module path"),
    recipe: str = typer.Option(..., "--recipe", help="Path to recipe YAML file"),
    input_shape: str = typer.Option(..., "--input-shape", help="Input shape, e.g. '1,3,224,224'"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Output directory"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Estimate only"),
    json_output: bool = typer.Option(False, "--json", help="Output report as JSON"),
):
    """Run a compression pipeline from a recipe."""
    from comprexx.recipe.loader import load_recipe, recipe_to_pipeline

    shape = _parse_input_shape(input_shape)
    model = _load_model(model_source)

    recipe_obj = load_recipe(recipe)
    pipeline, guard = recipe_to_pipeline(recipe_obj)

    with console.status("Compressing..."):
        result = pipeline.run(
            model,
            input_shape=shape,
            accuracy_guard=guard,
            dry_run=dry_run,
            output_dir=output_dir,
        )

    if json_output:
        console.print(result.report.to_json())
    else:
        console.print(Panel(result.report.summary(), title="Compression Report"))
        console.print(f"Artifacts saved to: {result.run_dir}")


@app.command(name="export")
def export_cmd(
    model_path: str = typer.Argument(..., help="Path to .pt model file"),
    format: str = typer.Option("onnx", "--format", help="Export format"),
    input_shape: str = typer.Option(..., "--input-shape", help="Input shape, e.g. '1,3,224,224'"),
    output_dir: str = typer.Option("./comprexx_artifacts", "--output-dir", help="Output dir"),
):
    """Export a model to deployment format."""
    shape = _parse_input_shape(input_shape)
    model = _load_model(model_path)

    if format == "onnx":
        from comprexx.export.onnx import ONNXExporter

        out_path = Path(output_dir) / "model.onnx"
        exporter = ONNXExporter()
        with console.status("Exporting to ONNX..."):
            exporter.export(model, input_shape=shape, output_path=str(out_path))
        console.print(f"[green]Exported to {out_path}[/green]")
        console.print(f"Manifest: {Path(output_dir) / 'comprexx_manifest.json'}")
    else:
        console.print(f"[red]Format '{format}' not supported in v0.1. Available: onnx[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
