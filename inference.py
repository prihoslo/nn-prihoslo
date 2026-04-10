"""Загрузка чекпоинта и инференс для MyEfficientNet."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.models import EfficientNet_B2_Weights

from bek_mode import MyEfficientNet


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_state_dict_keys(state: dict[str, Any]) -> dict[str, Any]:
    if not state:
        return state
    if any(k.startswith("module.") for k in state):
        return {k[len("module.") :]: v for k, v in state.items() if k.startswith("module.")}
    return state


def parse_checkpoint(weights_path: str | Path, map_location: torch.device) -> tuple[dict[str, Any], list[str] | None, int]:
    raw = torch.load(weights_path, map_location=map_location, weights_only=True)
    class_names: list[str] | None = None
    num_classes = 200
    if isinstance(raw, dict):
        class_names = raw.get("class_names")
        if class_names is not None:
            class_names = list(class_names)
        num_classes = int(raw.get("num_classes", num_classes))
        if "model_state_dict" in raw:
            state = raw["model_state_dict"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            sample = next((k for k in raw if isinstance(k, str)), None)
            if sample and (sample.startswith("model.") or sample.startswith("module.")):
                state = {k: v for k, v in raw.items() if isinstance(k, str) and not k.startswith("_")}
            else:
                state = raw.get("state_dict", raw)
    else:
        state = raw
    if not isinstance(state, dict):
        raise ValueError("Чекпоинт не содержит state_dict (ожидался dict с весами).")
    state = _normalize_state_dict_keys(state)
    return state, class_names, num_classes


def build_model_and_load(
    weights_path: str | Path,
    device: torch.device,
) -> tuple[MyEfficientNet, list[str] | None]:
    state, class_names, num_classes = parse_checkpoint(weights_path, device)
    model = MyEfficientNet(num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, class_names


def default_transforms():
    return EfficientNet_B2_Weights.DEFAULT.transforms()


def predict_one(
    pil_image: Image.Image,
    model: MyEfficientNet,
    device: torch.device,
    transforms,
    top_k: int = 5,
) -> tuple[list[int], list[float], float]:
    """Возвращает индексы top-k, вероятности top-k, время только forward (сек)."""
    tensor = transforms(pil_image.convert("RGB")).unsqueeze(0).to(device, non_blocking=True)
    with torch.inference_mode():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    probs = torch.softmax(logits[0], dim=-1)
    k = min(top_k, probs.numel())
    top_p, top_i = torch.topk(probs, k=k)
    return top_i.cpu().tolist(), top_p.cpu().tolist(), elapsed


def load_class_names_from_file(path: str | Path) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines
