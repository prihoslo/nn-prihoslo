"""Классификация изображений: URL, несколько файлов, время инференса."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
import torch
from PIL import Image

from inference import (
    build_model_and_load,
    default_transforms,
    get_torch_device,
    load_class_names_from_file,
    predict_one,
)

ROOT = Path(__file__).resolve().parent
REQUEST_TIMEOUT = 20

WEIGHTS_PATH = ROOT / "../efficientnet_b2_200cls.pth"
CLASS_NAMES_TXT_PATH: str | None = None
TOP_K = 5


@st.cache_resource
def cached_model_bundle(weights_path_str: str) -> tuple:
    path = Path(weights_path_str)
    device = get_torch_device()
    model, class_names_from_ckpt = build_model_and_load(path, device)
    transforms = default_transforms()
    return model, transforms, device, class_names_from_ckpt


def resolve_class_names(names_from_ckpt: list[str] | None, path_override: str | None) -> tuple[list[str] | None, str | None]:
    if path_override and path_override.strip():
        p = Path(path_override.strip()).expanduser()
        if p.is_file():
            return load_class_names_from_file(p), f"Файл: `{p}`"
        return names_from_ckpt, f"Файл не найден: `{p}`, используются метки из чекпоинта."
    return names_from_ckpt, None


def label_for_index(class_names: list[str] | None, idx: int) -> str:
    if class_names and 0 <= idx < len(class_names):
        return class_names[idx]
    return f"Класс {idx}"


def fetch_image_from_url(url: str) -> Image.Image:
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Разрешены только ссылки http и https.")
    headers = {"User-Agent": "StreamlitImageClassifier/1.0"}
    r = requests.get(url.strip(), timeout=REQUEST_TIMEOUT, headers=headers)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


st.set_page_config(
    page_title="Классификация изображений",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Классификация изображений")

load_error: str | None = None
model = transforms = device = None
names_from_ckpt: list[str] | None = None
try:
    model, transforms, device, names_from_ckpt = cached_model_bundle(str(WEIGHTS_PATH))
except Exception as e:
    load_error = str(e)

class_names, _ = resolve_class_names(names_from_ckpt, CLASS_NAMES_TXT_PATH)

if load_error:
    st.error(f"Не удалось загрузить модель: {load_error}")
    st.stop()

st.caption(f"Устройство: `{device}` · PyTorch `{torch.__version__}`")

tab_url, tab_files = st.tabs(["По ссылке", "Загрузка файлов"])

with tab_url:
    url = st.text_input("URL изображения", placeholder="https://example.com/image.jpg")
    run_url = st.button("Классифицировать по ссылке", type="primary")
    if run_url:
        if not url.strip():
            st.warning("Вставьте ссылку на изображение.")
        else:
            try:
                img = fetch_image_from_url(url)
            except Exception as e:
                st.error(f"Не удалось загрузить изображение: {e}")
            else:
                idxs, probs, elapsed = predict_one(img, model, device, transforms, top_k=TOP_K)
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.image(img, caption="Изображение по ссылке", use_container_width=True)
                with c2:
                    st.metric("Время инференса модели", f"{elapsed:.4f} с")
                    lines = [
                        f"{i + 1}. **{label_for_index(class_names, idxs[i])}** — {probs[i] * 100:.2f}%"
                        for i in range(len(idxs))
                    ]
                    st.markdown("**Предсказания:**\n\n" + "\n\n".join(lines))

with tab_files:
    uploaded = st.file_uploader(
        "Выберите одно или несколько изображений",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    if uploaded:
        timings: list[dict[str, float | str]] = []
        for f in uploaded:
            try:
                data = f.getvalue()
                img = Image.open(BytesIO(data)).convert("RGB")
            except Exception as e:
                st.error(f"{f.name}: не удалось прочитать файл — {e}")
                continue
            idxs, probs, elapsed = predict_one(img, model, device, transforms, top_k=TOP_K)
            timings.append({"Файл": f.name, "Секунды": elapsed})
            st.subheader(f.name)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.image(img, caption=f.name, use_container_width=True)
            with c2:
                st.metric("Время инференса", f"{elapsed:.4f} с")
                lines = [
                    f"{i + 1}. **{label_for_index(class_names, idxs[i])}** — {probs[i] * 100:.2f}%"
                    for i in range(len(idxs))
                ]
                st.markdown("**Предсказания:**\n\n" + "\n\n".join(lines))
            st.divider()

        if timings:
            df = pd.DataFrame(timings)
            st.subheader("Время ответа модели по файлам")
            mean_s = float(df["Секунды"].mean())
            st.metric("Среднее время инференса", f"{mean_s:.4f} с")
            st.bar_chart(df, x="Файл", y="Секунды", horizontal=True)
    else:
        st.info("Загрузите изображения для классификации.")
