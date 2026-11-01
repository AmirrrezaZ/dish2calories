"""
Shared model-loading and result-formatting helpers.

Kept free of FastAPI/Streamlit imports so both the API (inference.py) and the
Streamlit app (app.py, in-process mode) can use it with a light dependency set.
"""
import os
from pathlib import Path
from typing import Dict

import torch

from dataset_loaders import DatasetInfo
from models import CalorieNet
from predict import CalorieNetPredictor

MODEL_TYPE = "efficientnet_b0"
DATASET_CONFIG = "configs/dataset_info.json"
WEIGHTS_DIR = "weights"
CLS_THRESHOLD = 0.5
TEMPERATURE = 0.5
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def load_predictor() -> CalorieNetPredictor:
    """Load weights + class info and build the predictor."""
    config_path = Path(DATASET_CONFIG)
    weights_path = Path(WEIGHTS_DIR) / f"{MODEL_TYPE}.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    info = DatasetInfo.from_json(str(config_path))
    model, _ = CalorieNet.from_checkpoint(str(weights_path), device=DEVICE)
    return CalorieNetPredictor(
        model=model,
        norm_ingr=info.norm_ingr,
        scaler=info.scalers["total_mass"],
        idx2cls=info.idx2cls,
        img_size=info.img_size,
        cls_threshold=CLS_THRESHOLD,
        temperature=TEMPERATURE,
        mode="probs",
        device=DEVICE,
    )


def format_results(results: Dict) -> Dict:
    """Convert raw predictor output into a JSON-friendly dict."""
    return {
        "total_protein": float(results["total_protein"][0]),
        "total_fat": float(results["total_fat"][0]),
        "total_carbs": float(results["total_carbs"][0]),
        "total_calories": float(results["total_calories"][0]),
        "total_mass": float(results["total_mass"][0]),
        "labels_conf": [float(c) for c in results["labels_conf"]],
        "mean_conf": float(results["mean_conf"]),
        "labels_txt": results["labels_txt"],
    }
