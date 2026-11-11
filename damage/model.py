"""Portion 1 — Damage Segmentation Model Definition and Offline Checkpoint Loader."""

from __future__ import annotations

import logging
import pickle
import threading
import zipfile
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch
from torch import nn

_LOGGER = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = ROOT_DIR / "data" / "models" / "unet_resnet18_best.pth"

NUM_CLASSES = 5  # 0 background, 1 no_damage, 2 minor, 3 major, 4 destroyed
IGNORE_INDEX = 255
FEATURE_LAYER = "model.decoder (final decoder block output)"

CLASS_INDEX_TO_CANONICAL: dict[int, str] = {
    1: "no_damage",
    2: "minor",
    3: "major",
    4: "destroyed",
}

_LABEL_ALIASES: dict[str, str] = {
    "no_damage": "no_damage",
    "no-damage": "no_damage",
    "minor": "minor",
    "minor-damage": "minor",
    "major": "major",
    "major-damage": "major",
    "destroyed": "destroyed",
}

_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], nn.Module] = {}


def model_class_to_canonical(class_index: int) -> str:
    """Convert model segmentation class index (1..4) to Portion 5 canonical damage_class."""
    if isinstance(class_index, bool) or class_index not in CLASS_INDEX_TO_CANONICAL:
        raise ValueError(
            f"Model class index must be in 1..4 (building classes), got {class_index!r}"
        )
    return CLASS_INDEX_TO_CANONICAL[class_index]


def normalize_damage_label(raw: Any) -> str:
    """Normalize xBD hyphenated or canonical strings into Portion 5 damage labels."""
    text = str(raw).strip().lower()
    if text not in _LABEL_ALIASES:
        raise ValueError(f"Unsupported damage label: {raw!r}")
    return _LABEL_ALIASES[text]


class ResidualAdapterHead(nn.Module):
    """Lightweight 1x1 bottleneck residual adapter for Tier A frozen-encoder adaptation."""

    def __init__(self, dec_channels: int, num_classes: int = NUM_CLASSES, bottleneck: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dec_channels, bottleneck, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck, num_classes, kernel_size=1, bias=True),
        )
        # Zero-init final projection so un-trained adapter preserves base predictions exactly
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, dec_features: torch.Tensor) -> torch.Tensor:
        return self.net(dec_features)


class AdaptedSegmentationModel(nn.Module):
    """Wraps ResNet-18 U-Net preserving .encoder/.decoder hooks for Grad-CAM & embeddings."""

    def __init__(self, base_model: nn.Module, bottleneck: int = 16) -> None:
        super().__init__()
        self.base_model = base_model
        seg_first = base_model.segmentation_head[0] if hasattr(base_model.segmentation_head, "__getitem__") else None
        dec_channels = int(getattr(seg_first, "in_channels", 16))
        self.adapter_block = ResidualAdapterHead(dec_channels, NUM_CLASSES, bottleneck=bottleneck)
        self.adapter_scale = nn.Parameter(torch.tensor(1.0))

    @property
    def encoder(self) -> nn.Module:
        return self.base_model.encoder

    @property
    def decoder(self) -> nn.Module:
        return self.base_model.decoder

    @property
    def segmentation_head(self) -> nn.Module:
        return self.base_model.segmentation_head

    def freeze_backbone(self) -> None:
        """Freeze encoder, decoder, and base head weights/BN stats so Tier A trains only the adapter."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()

    def train(self, mode: bool = True) -> "AdaptedSegmentationModel":
        """Switch adapter head train/eval mode while keeping base backbone strictly in eval mode."""
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_feats = self.base_model.encoder(x)
        dec_out = self.base_model.decoder(enc_feats)
        base_logits = self.base_model.segmentation_head(dec_out)
        return base_logits + self.adapter_scale * self.adapter_block(dec_out)


def build_model(pretrained_encoder: bool = False) -> nn.Module:
    """Construct the 6-channel bi-temporal ResNet-18 U-Net segmentation architecture."""
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet" if pretrained_encoder else None,
        in_channels=6,
        classes=NUM_CLASSES,
    )


def resolve_device(device: str | None = None) -> str:
    """Resolve requested or best available compute device ('cuda' or 'cpu')."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_damage_model(
    weights_path: str | Path | None = None,
    device: str | None = None,
    use_cache: bool = True,
    adapter_path: str | Path | None = None,
) -> tuple[nn.Module, str]:
    """Load pretrained ResNet-18 U-Net weights offline (plus optional promoted adapter) in eval mode."""
    ckpt = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not ckpt.is_file():
        raise FileNotFoundError(f"Pretrained damage weights not found: {ckpt}")

    adapter_ckpt = Path(adapter_path) if adapter_path is not None else None
    has_adapter = adapter_ckpt is not None and adapter_ckpt.is_file()
    target_device = resolve_device(device)
    cache_key = (f"{ckpt.resolve()}::{adapter_ckpt.resolve() if has_adapter else 'none'}", target_device)

    if use_cache:
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached, target_device

    net = build_model(pretrained_encoder=False)
    state_dict = torch.load(ckpt, map_location=target_device, weights_only=True)
    net.load_state_dict(state_dict, strict=True)

    adapter_loaded = False
    if has_adapter and adapter_ckpt is not None:
        try:
            payload = torch.load(adapter_ckpt, map_location=target_device, weights_only=True)
            if not isinstance(payload, dict) or "adapter_state_dict" not in payload:
                raise ValueError(f"Invalid adapter checkpoint payload in {adapter_ckpt}")
            bottleneck = max(1, int(payload.get("bottleneck", 16)))
            adapted = AdaptedSegmentationModel(net, bottleneck=bottleneck)
            adapted.adapter_block.load_state_dict(payload["adapter_state_dict"], strict=True)
            if "adapter_scale" in payload:
                with torch.no_grad():
                    adapted.adapter_scale.copy_(payload["adapter_scale"])
            net = adapted
            adapter_loaded = True
        except (
            OSError,
            RuntimeError,
            ValueError,
            EOFError,
            KeyError,
            TypeError,
            pickle.PickleError,
            zipfile.BadZipFile,
        ) as exc:
            _LOGGER.warning("Failed to load adapter %s (%s); falling back to base model weights.", adapter_ckpt, exc)

    net.to(target_device).eval()

    if use_cache and (not has_adapter or adapter_loaded):
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = net

    return net, target_device

