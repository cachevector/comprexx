"""ONNX model export with validation and manifest generation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from comprexx.core.exceptions import ExportError
from comprexx.core.report import CompressionReport
from comprexx.export.manifest import ExportManifest, compute_model_hash


class ONNXExporter:
    """Export PyTorch models to ONNX format with validation."""

    def __init__(
        self,
        opset_version: int = 17,
        simplify: bool = True,
        validate: bool = True,
    ):
        self.opset_version = opset_version
        self.simplify = simplify
        self.validate = validate

    def export(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...],
        output_path: str,
        dynamic_axes: Optional[dict] = None,
        compression_report: Optional[CompressionReport] = None,
    ) -> ExportManifest:
        """Export model to ONNX format.

        Args:
            model: PyTorch model to export.
            input_shape: Input tensor shape (including batch dim).
            output_path: Path for the .onnx output file.
            dynamic_axes: ONNX dynamic axes specification.
            compression_report: Optional report to include in manifest.

        Returns:
            ExportManifest with export metadata.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        dummy_input = torch.randn(*input_shape)

        # Get PyTorch output for validation
        with torch.no_grad():
            try:
                pt_output = model(dummy_input)
            except Exception as e:
                raise ExportError(f"Model forward pass failed: {e}") from e

        # Export to ONNX
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                opset_version=self.opset_version,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
        except Exception as e:
            raise ExportError(f"ONNX export failed: {e}") from e

        # Simplify
        if self.simplify:
            self._try_simplify(output_path)

        # Validate
        if self.validate:
            self._validate(output_path, dummy_input, pt_output)

        # Build manifest
        model_hash = compute_model_hash(model)
        compression_stats = {}
        if compression_report is not None:
            compression_stats = {
                "size_reduction_pct": compression_report.total_size_reduction_pct,
                "flops_reduction_pct": compression_report.total_flops_reduction_pct,
                "compression_ratio": compression_report.total_compression_ratio,
            }

        manifest = ExportManifest(
            original_model_name=model.__class__.__name__,
            original_model_hash=model_hash,
            export_format="onnx",
            export_format_version=f"opset_{self.opset_version}",
            compression_stats=compression_stats,
        )

        manifest_path = output_path.parent / "comprexx_manifest.json"
        manifest.save(manifest_path)

        return manifest

    def _try_simplify(self, path: Path) -> None:
        """Try to simplify the ONNX model using onnxsim."""
        try:
            import onnx
            import onnxsim

            model = onnx.load(str(path))
            simplified, ok = onnxsim.simplify(model)
            if ok:
                onnx.save(simplified, str(path))
        except ImportError:
            pass  # onnxsim not installed, skip
        except Exception:
            pass  # Simplification is best-effort

    def _validate(
        self, path: Path, dummy_input: torch.Tensor, pt_output: torch.Tensor
    ) -> None:
        """Validate ONNX model output matches PyTorch output."""
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(path))
            input_name = session.get_inputs()[0].name
            ort_output = session.run(None, {input_name: dummy_input.numpy()})

            pt_np = pt_output.detach().numpy()
            max_diff = abs(pt_np - ort_output[0]).max()

            if max_diff > 1e-5:
                warnings.warn(
                    f"ONNX output differs from PyTorch by {max_diff:.6f} "
                    f"(threshold: 1e-5). This may indicate numerical issues.",
                    stacklevel=2,
                )
        except ImportError:
            pass  # onnxruntime not installed, skip validation
        except Exception as e:
            warnings.warn(
                f"ONNX validation failed: {e}",
                stacklevel=2,
            )
