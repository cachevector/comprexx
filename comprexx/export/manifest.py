"""Export manifest — metadata for compressed model artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch.nn as nn

import comprexx


@dataclass
class ExportManifest:
    """Metadata describing an exported model artifact."""

    original_model_name: str
    original_model_hash: str
    export_format: str
    export_format_version: str = ""
    compression_stats: dict = field(default_factory=dict)
    recipe_used: Optional[dict] = None
    target_hardware: Optional[str] = None
    comprexx_version: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.comprexx_version:
            self.comprexx_version = getattr(comprexx, "__version__", "0.1.0")
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def compute_model_hash(model: nn.Module) -> str:
    """Compute a SHA256 hash of the model's state dict."""
    h = hashlib.sha256()
    for key, param in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(param.cpu().numpy().tobytes())
    return h.hexdigest()[:16]
