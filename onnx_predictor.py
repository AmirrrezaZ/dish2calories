"""
Lightweight CalorieNet inference using ONNX Runtime (no PyTorch needed).

Produces the same output dict as ``CalorieNetPredictor.predict_img`` but only depends on
numpy, Pillow and onnxruntime (~20 MB instead of ~200 MB-2.5 GB for torch), which makes
Streamlit Cloud installs and cold starts much faster.

Export the model first with:  python export_onnx.py
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


class OnnxCalorieNetPredictor:
    def __init__(
        self,
        onnx_path: Union[str, Path],
        config_path: Union[str, Path],
        cls_threshold: float = 0.5,
        temperature: float = 0.5,
        mode: str = "probs",
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            info = json.load(f)

        self.img_size = int(info.get("img_size", 256))
        self.norm_ingr = np.asarray(info["norm_ingr"], dtype=np.float32)  # [C, D]
        idx2cls = {int(k): v for k, v in info["idx2cls"].items()}
        self.idx2cls: List[str] = [idx2cls[i] for i in range(len(idx2cls))]

        sc = (info.get("scalers") or {}).get("total_mass")
        if sc is not None:
            self.mass_mean = float(sc["mean_"][0]) if sc.get("with_mean", True) else 0.0
            self.mass_scale = float(sc["scale_"][0]) if sc.get("with_std", True) else 1.0
        else:
            self.mass_mean, self.mass_scale = 0.0, 1.0

        self.cls_threshold = float(cls_threshold)
        self.temperature = float(temperature)
        self.mode = mode

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    # ---------------- preprocessing ----------------
    def _preprocess(self, pil: Image.Image) -> np.ndarray:
        pil = pil.convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        x = np.asarray(pil, dtype=np.float32) / 255.0
        x = (x - _MEAN) / _STD
        return x.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,H,W]

    # ---------------- postprocessing (mirrors predict.py) ----------------
    def _per_gram(self, probs: np.ndarray) -> np.ndarray:
        mask = (probs >= self.cls_threshold).astype(np.float32)
        if self.mode == "binary":
            weights = mask / np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
            return weights @ self.norm_ingr
        scores = np.clip(np.log(np.clip(probs, 1e-12, None) / max(self.temperature, 1e-6)), -20, None)
        scores = scores + (mask - 1.0) * 1e9
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores) * mask
        denom = np.clip(exp_scores.sum(axis=1, keepdims=True), 1.0, None)
        return (exp_scores / denom) @ self.norm_ingr

    def predict_img(self, img: Union[str, Path, np.ndarray, Image.Image]) -> Dict[str, Any]:
        if isinstance(img, (str, Path)):
            pil = Image.open(img)
        elif isinstance(img, Image.Image):
            pil = img
        elif isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Numpy array must be HxWx3 RGB.")
            arr = img if img.dtype == np.uint8 else (np.clip(img, 0, 1) * 255).round().astype(np.uint8)
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError("img must be a path, PIL image, or HxWx3 RGB numpy array.")

        logits, mass = self.session.run(["classification", "mass"], {"image": self._preprocess(pil)})
        probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        probs = probs.astype(np.float32)
        preds = (probs >= self.cls_threshold).astype(np.int32)

        idxs = np.nonzero(preds[0])[0].tolist()
        confs = [float(probs[0, i]) for i in idxs]

        per_gram = self._per_gram(probs)  # [1, D]
        mass_g = np.clip(mass.reshape(-1).astype(np.float32) * self.mass_scale + self.mass_mean, 0.0, None)

        prot, fat, carb = per_gram[:, 0] * mass_g, per_gram[:, 1] * mass_g, per_gram[:, 2] * mass_g
        return {
            "total_protein": prot,
            "total_fat": fat,
            "total_carbs": carb,
            "total_calories": 4.0 * prot + 9.0 * fat + 4.0 * carb,
            "total_mass": mass_g,
            "per_gram_macros": per_gram,
            "preds_bin": preds,
            "labels_idx": idxs,
            "labels_conf": confs,
            "mean_conf": float(np.mean(confs)) if confs else 0.0,
            "labels_txt": [self.idx2cls[i] for i in idxs],
        }
