from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class CalorieNetPredictor:
    """
    Public API:
      - predict_dataset(dataloader)
      - predict_batch(batch)
      - predict_img(img)          # img: str|Path to image OR HxWx3 RGB numpy array

    Assumptions
    -----------
    - model(imgs) -> {
        "classification": logits [B, C],
        "mass":           [B] or [B,1]  (normalized if scaler provided)
      }
    - norm_ingr: [C, D] per-gram macro features (e.g., [protein, fat, carb, ...]).
    - scaler: sklearn-like with inverse_transform for mass, or None.
    """

    def __init__(
        self,
        model: nn.Module,
        norm_ingr: Union[torch.Tensor, np.ndarray],
        scaler: Optional[Any] = None,                  # sklearn-like scaler or None
        idx2cls: Optional[List[str]] = None,           # optional class-name mapping
        img_size: int = 224,
        device: Union[str, torch.device] = None,
        cls_threshold: float = 0.5,
        temperature: float = 0.5,
        mode: Literal["binary", "probs"] = "probs",
    ):
        # --- device & model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval() 
        # --- static data
        self.scaler = scaler
        self.idx2cls = idx2cls
        self.cls_threshold = float(cls_threshold)
        self.img_size = int(img_size)
        self.temperature = float(temperature)
        self.mode: Literal["binary", "probs"] = mode

        if not torch.is_tensor(norm_ingr):
            norm_ingr = torch.as_tensor(norm_ingr)
        if norm_ingr.ndim != 2:
            raise ValueError("norm_ingr must be 2D [C, D].")
        self.norm_ingr = norm_ingr.to(self.device, dtype=torch.float32)  # [C, D]

        # torchvision preprocessing
        self._transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # -------------------- internals --------------------
    @torch.no_grad()
    def _predict_nutrition_from_classification(self, preds_bin: torch.Tensor) -> torch.Tensor:
        """
        preds_bin: [B, C] in {0,1}
        returns : [B, D] per-gram nutrients (weighted AVERAGE of selected components).
        """
        if preds_bin.ndim != 2:
            raise ValueError("preds_bin must be [B, C].")
        if preds_bin.shape[1] != self.norm_ingr.shape[0]:
            raise ValueError(f"preds_bin C={preds_bin.shape[1]} != norm_ingr C={self.norm_ingr.shape[0]}")
    
        # Equal weight over positives (fallback to zeros if no positive)
        counts = preds_bin.sum(dim=1, keepdim=True)                  # [B,1]
        weights = preds_bin.float() / counts.clamp_min(1.0)          # [B,C]
        # For rows with no positives, weights are all 0 (since preds_bin==0); result will be 0
        return weights @ self.norm_ingr                               # [B,D] per-gram
    
    @torch.no_grad()
    def _predict_nutrition_from_probs(self, probs: torch.Tensor, thresh: float = 0.5, temperature: float = 1.0) -> torch.Tensor:
        """
        probs: [B,C] in [0,1] (e.g., sigmoid(logits))
        Uses a masked softmax over active classes to get composition weights.
        """
        if probs.ndim != 2 or probs.shape[1] != self.norm_ingr.shape[0]:
            raise ValueError("probs shape must be [B, C] matching norm_ingr.")
    
        mask = (probs >= thresh).float()                               # [B,C]
        # masked softmax (numerically stable)
        scores = (probs / max(temperature, 1e-6)).log().clamp_min(-20) # log to focus high probs; clamp for stability
        scores = scores + (mask - 1.0) * 1e9                           # -inf where mask==0
        scores = scores - scores.max(dim=1, keepdim=True).values
        exp_scores = (scores.exp() * mask)
        denom = exp_scores.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = exp_scores / denom                                   # rows with no actives → weights all 0
        return weights @ self.norm_ingr                                 # [B,D] per-gram


    @staticmethod
    def _calories_from_macros(prot_g: torch.Tensor, fat_g: torch.Tensor, carb_g: torch.Tensor) -> torch.Tensor:
        return 4.0 * prot_g + 9.0 * fat_g + 4.0 * carb_g

    @torch.no_grad()
    def _postprocess_classification(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, List[List[int]], List[List[float]], List[float]]:
        if logits.ndim != 2:
            raise ValueError("classification logits must be [B, C].")
        probs = torch.sigmoid(logits)
        preds = (probs >= self.cls_threshold).int()

        B, _ = preds.shape
        labels_batch: List[List[int]] = [[] for _ in range(B)]
        confs_batch: List[List[float]] = [[] for _ in range(B)]

        pos = (preds == 1).nonzero(as_tuple=False)
        for r, c in pos.tolist():
            labels_batch[r].append(c)
            confs_batch[r].append(float(probs[r, c].item()))

        conf_scalar = [float(np.mean(cs)) if cs else 0.0 for cs in confs_batch]
        return preds, probs, labels_batch, confs_batch, conf_scalar

    @torch.no_grad()
    def _inverse_mass(self, mass_pred: torch.Tensor) -> torch.Tensor:
        """
        mass_pred: [B] or [B,1] on device -> returns grams [B] on device.
        """
        mass_pred = mass_pred.squeeze(-1).float()
        if self.scaler is None:
            return mass_pred
        m = mass_pred.detach().cpu().numpy().reshape(-1, 1)
        m = self.scaler.inverse_transform(m).astype(np.float32).reshape(-1)
        m = np.clip(m, 0.0, None)
        return torch.from_numpy(m).to(self.device)

    @torch.no_grad()
    def _postprocess_batch(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        if "classification" not in outputs or "mass" not in outputs:
            raise KeyError("model outputs must contain 'classification' and 'mass'.")

        logits = outputs["classification"]
        mass_pred = outputs["mass"]

        preds_bin, probs, labels_idx, confs, conf_scalar = self._postprocess_classification(logits)

        # --- per-gram composition by configured mode
        if self.mode == "binary":
            per_gram = self._predict_nutrition_from_classification(preds_bin)
        else:  # "probs"
            per_gram = self._predict_nutrition_from_probs(
                probs, thresh=self.cls_threshold, temperature=self.temperature
            )
        
        
        
        if per_gram.shape[1] < 3:
            raise ValueError("norm_ingr must have at least 3 columns [protein, fat, carb] per gram.")

        prot_pg, fat_pg, carb_pg = per_gram[:, 0], per_gram[:, 1], per_gram[:, 2]
        mass_g = self._inverse_mass(mass_pred)  # [B]

        prot_total = prot_pg * mass_g
        fat_total  = fat_pg  * mass_g
        carb_total = carb_pg * mass_g
        cal_total  = self._calories_from_macros(prot_total, fat_total, carb_total)

        out: Dict[str, Any] = {
            "total_protein": prot_total.detach().cpu().numpy(),
            "total_fat":     fat_total.detach().cpu().numpy(),
            "total_carbs":   carb_total.detach().cpu().numpy(),
            "total_calories":cal_total.detach().cpu().numpy(),
            "total_mass":    mass_g.detach().cpu().numpy(),
            "per_gram_macros": per_gram.detach().cpu().numpy(),   # [B, D]
            "labels_idx":    labels_idx,                          # List[List[int]]
            "labels_conf":   confs,                               # List[List[float]]
            "mean_conf":     conf_scalar,                         # List[float]
            "preds_bin":     preds_bin.detach().cpu().numpy(),    # [B, C]
        }
        if self.idx2cls is not None:
            out["labels_txt"] = [[self.idx2cls[j] for j in idxs] for idxs in labels_idx]
        return out

    # -------------------- public API --------------------

    @torch.no_grad()
    def predict_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        batch: dict with key "image" -> [B, 3, H, W] float32 (0–1 or arbitrary scale; transform should have normalized already upstream if needed).
        """
        images = batch["image"].to(self.device, dtype=torch.float32)
        outputs = self.model(images)
        
        out = self._postprocess_batch(outputs)
        out["img_path"] = batch['img_path']
        
        return out

    @torch.no_grad()
    def predict_dataset(
        self,
        data_loader: DataLoader,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        self.model.eval()
        agg: Dict[str, List[Any]] = {
            "total_protein": [], "total_fat": [], "total_carbs": [],
            "total_calories": [], "total_mass": [], "per_gram_macros": [],
            "preds_bin": [], "labels_idx": [], "labels_conf": [], "mean_conf": [], 'img_path':[],
        }
        if self.idx2cls is not None:
            agg["labels_txt"] = []

        iterator = tqdm(data_loader, total=len(data_loader), desc="Evaluating", disable=not show_progress)
        for batch in iterator:
            batch_out = self.predict_batch(batch)

            for k in ["total_protein","total_fat","total_carbs","total_calories","total_mass"]:
                agg[k].append(batch_out[k])
            agg["per_gram_macros"].append(batch_out["per_gram_macros"])
            agg["preds_bin"].append(batch_out["preds_bin"])
            agg["labels_idx"].extend(batch_out["labels_idx"])
            agg["labels_conf"].extend(batch_out["labels_conf"])
            agg["mean_conf"].extend(batch_out["mean_conf"])
            agg["img_path"].extend(batch_out.get("img_path", []))
            if "labels_txt" in batch_out:
                agg["labels_txt"].extend(batch_out["labels_txt"])

        final: Dict[str, Any] = {
            "total_protein":  np.concatenate(agg["total_protein"], axis=0) if agg["total_protein"] else np.array([]),
            "total_fat":      np.concatenate(agg["total_fat"], axis=0) if agg["total_fat"] else np.array([]),
            "total_carbs":    np.concatenate(agg["total_carbs"], axis=0) if agg["total_carbs"] else np.array([]),
            "total_calories": np.concatenate(agg["total_calories"], axis=0) if agg["total_calories"] else np.array([]),
            "total_mass":     np.concatenate(agg["total_mass"], axis=0) if agg["total_mass"] else np.array([]),
            "per_gram_macros":np.concatenate(agg["per_gram_macros"], axis=0) if agg["per_gram_macros"] else np.empty((0, self.norm_ingr.shape[1]), dtype=np.float32),
            "preds_bin":      np.concatenate(agg["preds_bin"], axis=0) if agg["preds_bin"] else np.empty((0, self.norm_ingr.shape[0]), dtype=np.int32),
            "labels_idx":     agg["labels_idx"],
            "labels_conf":    agg["labels_conf"],
            "mean_conf":      agg["mean_conf"],
            "img_path":       agg["img_path"],
        }
        if "labels_txt" in agg:
            final["labels_txt"] = agg["labels_txt"]
        return final

    @torch.no_grad()
    def predict_img(self, img: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        img: file path or HxWx3 RGB numpy array (uint8 [0..255] or float [0..1]).
        Returns the same dict structure as a 1-sample batch.
        """

        img_path_str = ""
        # --- load to PIL
        if isinstance(img, (str, Path)):
            img_path_str = str(img)
            pil = Image.open(img_path_str).convert("RGB")
        elif isinstance(img, np.ndarray):
            if img.ndim != 3 or img.shape[2] != 3:
                raise ValueError("Numpy array must be HxWx3 RGB.")
            if img.dtype != np.uint8:
                arr = np.clip(img, 0.0, 1.0)
                arr = (arr * 255.0).round().astype(np.uint8)
            else:
                arr = img
            pil = Image.fromarray(arr, mode="RGB")
        else:
            raise TypeError("img must be a path or an HxWx3 RGB numpy array.")    
            

        # --- preprocess
        x = self._transform(pil).unsqueeze(0).to(self.device, dtype=torch.float32)  # [1,3,H,W]

        # --- model forward
        outputs = self.model(x)
        out = self._postprocess_batch(outputs)

        # squeeze arrays to 1D for single sample
        def _squeeze(v):
            v = np.asarray(v)
            return v.reshape(-1) if v.ndim > 1 else v

        single = {
            "total_protein":  _squeeze(out["total_protein"]),
            "total_fat":      _squeeze(out["total_fat"]),
            "total_carbs":    _squeeze(out["total_carbs"]),
            "total_calories": _squeeze(out["total_calories"]),
            "total_mass":     _squeeze(out["total_mass"]),
            "per_gram_macros": out["per_gram_macros"].reshape(1, -1) if out["per_gram_macros"].ndim == 2 else out["per_gram_macros"],
            "preds_bin":      out["preds_bin"].reshape(1, -1) if out["preds_bin"].ndim == 2 else out["preds_bin"],
            "labels_idx":     out["labels_idx"][0] if out["labels_idx"] else [],
            "labels_conf":    out["labels_conf"][0] if out["labels_conf"] else [],
            "mean_conf":      float(out["mean_conf"][0]) if out["mean_conf"] else 0.0,
            "img_path":       [img_path_str] if img_path_str else [""],
        }
        if "labels_txt" in out:
            single["labels_txt"] = out["labels_txt"][0] if out["labels_txt"] else []
        return single
