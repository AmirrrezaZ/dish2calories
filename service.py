"""
Shared model-loading and result-formatting helpers.

Kept free of FastAPI/Streamlit imports so both the API (inference.py) and the
Streamlit app (app.py, in-process mode) can use it with a light dependency set.
"""
from pathlib import Path
from typing import Dict

from onnx_predictor import OnnxCalorieNetPredictor

MODEL_TYPE = "efficientnet_b0"
DATASET_CONFIG = "configs/dataset_info.json"
WEIGHTS_DIR = "weights"
CLS_THRESHOLD = 0.5
TEMPERATURE = 0.5


def load_predictor() -> OnnxCalorieNetPredictor:
    """Load the ONNX model (CPU, no PyTorch needed). Create it with `python export_onnx.py`."""
    config_path = Path(DATASET_CONFIG)
    onnx_path = Path(WEIGHTS_DIR) / f"{MODEL_TYPE}.onnx"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path} (run `python export_onnx.py`)")
    return OnnxCalorieNetPredictor(
        onnx_path, config_path, cls_threshold=CLS_THRESHOLD, temperature=TEMPERATURE, mode="probs"
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
