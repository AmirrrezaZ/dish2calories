import ast
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union

import numpy as np
import pandas as pd
import torch


from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import json
from sklearn.preprocessing import StandardScaler


# Handle truncated/corrupt images more gracefully
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import (
    scale_nutrition_features,
    build_metadata,
    stratified_split,
    build_classes)


class Nutrition5kDataset(Dataset):
    """
    Dataset for Nutrition5k images + multi-hot ingredient labels + nutrition targets.

    """


    def __init__(
        self,
        base_dir: Union[str, Path],
        n: int = 1,
        seed: int = 42,
        test_frac: float = 0.2,
        val_frac: float = 0.1,
        img_size: int = 224,
        split: str = "train",
        cols_to_scale : Optional[List[str]] = None,
        transform: Optional[Callable] = None,

    ):
        # -------- build df & splits --------
        base_dir = Path(base_dir)
        df = build_metadata(base_dir=str(base_dir), id_col="id", n=n, seed=seed)
 
        df = stratified_split(df, label_col="label", test_frac=test_frac, val_frac=val_frac, seed=seed)
        
        df_unscaled = df.copy()
        # --------StandardScaler --------
        df, scalers = scale_nutrition_features(df, scalers=None, cols = cols_to_scale)
    

        classes, cls2idx, idx2cls = build_classes(df, "label")
        weights = compute_class_weights(df, cls2idx, label_col="label", method="balanced")
        

        
        # -------- keep only the requested split --------
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split name: {split!r}. Expected one of ['train','val','test'].")

        df_split = df[df["tag"] == split].reset_index(drop=True)
        df_unscaled = df_unscaled[df_unscaled["tag"] == split].reset_index(drop=True)
        
        

        
        # -------- build portion free ingredients values --------
        norm_ingr_df = pd.read_csv(f"{str(base_dir)}/ingredients_metadata.csv")


        norm_ingr_dict = {
            str(row['ingr']).strip().lower(): {
                'protein': float(row['protein(g)']),
                'fat':     float(row['fat(g)']),
                'carbs':   float(row['carb(g)']),
            }
            for _, row in norm_ingr_df.iterrows()
        }

        class_to_idx_norm = {str(k).strip().lower(): v for k, v in cls2idx.items()}

        norm_ingr = build_norm_ingr_table(class_to_idx_norm, norm_ingr_dict)
        
  
        # -------- store state --------
        self.df = df_split
        self.split = split
        self.scalers = scalers
        self.cls2idx = cls2idx
        self.idx2cls = idx2cls
        self.num_classes = len(cls2idx)
        self.img_size = img_size
        self.class_weights = weights
        self.norm_ingr = norm_ingr
        self.transform = transform or transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        df_unscaled["cls_multi_hot"] = df_unscaled["label"].apply(self._encode_multi_hot)
        self.df_unscaled = df_unscaled
        self.df['cls_multi_hot'] = df_unscaled["cls_multi_hot"]


    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load image with robust fallback
        img_path = row["img_path"]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Corrupted/missing image -> create a black fallback
            print(f"[Nutrition5kDataset] Error loading image {img_path}: {e}")
            image = Image.new("RGB", (self.img_size, self.img_size), color="black")

        image = self.transform(image) if self.transform else image

        # --- Multi-hot labels ---
        # multi_hot = self._encode_multi_hot(row["label"])
        multi_hot = row['cls_multi_hot']

        # --- Nutrition features as tensors ---

        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        fat = torch.tensor(row['total_fat'], dtype=torch.float32)
        carb = torch.tensor(row['total_carb'], dtype=torch.float32)
        protein = torch.tensor(row['total_protein'], dtype=torch.float32)
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)


        sample = {
            "image": image,
            "cls_multi_hot": multi_hot,
            "label": row["label"],
            "mass": mass,
            "fat": fat,
            "carb": carb,
            "protein": protein,
            "calories": calories,
            "img_path": img_path,
        }
        return sample

    def _encode_multi_hot(self, labels_val: Union[str, List[str]]) -> torch.Tensor:
        """
        Accepts either a JSON-like string (e.g. "['egg','flour']") or a list of strings.
        """
        y = torch.zeros(self.num_classes, dtype=torch.float32)
        labels_list: List[str] = []

        if isinstance(labels_val, list):
            labels_list = labels_val
        elif isinstance(labels_val, str):
            try:
                parsed = ast.literal_eval(labels_val)
                if isinstance(parsed, list):
                    labels_list = parsed
                else:
                    # fallback: single label string
                    labels_list = [labels_val]
            except Exception:
                labels_list = [labels_val]
        else:
            # unknown format -> return all zeros
            return y

        for lab in labels_list:
            idx = self.cls2idx.get(lab)
            if idx is not None:
                y[idx] = 1.0
        return y

    def __repr__(self) -> str:
        return (
            f"Nutrition5kDataset(split={self.split!r}, n={len(self)}, "
            f"num_classes={self.num_classes}, img_size={self.img_size})"
        )




class Nutrition5kLoaders:
    """
    Thin convenience wrapper to build dataset + dataloader with consistent transforms.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = dict(config)  # shallow copy so we can .get safely
        self.img_size = int(self.config["img_size"])
        self.ds: Dict[str, Optional["Nutrition5kDataset"]] = {"train": None, "val": None, "test": None}

    def get_transforms(self, split: str) -> transforms.Compose:
        """
        Returns a torchvision image transformation pipeline based on the split.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split name: {split!r}. Expected one of ['train','val','test'].")

        is_train = split == "train"

        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=30),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                    ),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def get_dataset(self, split: str) -> "Nutrition5kDataset":
        """Create and return the requested split dataset (cached)."""
        if split not in self.ds:
            raise ValueError(f"Invalid split: {split!r}. Expected one of {list(self.ds.keys())}.")

        # If not cached, build and store
        if self.ds[split] is None:
            self.ds[split] = Nutrition5kDataset(
                base_dir=self.config["base_dir"],
                img_size=self.config["img_size"],
                split=split,
                n=self.config.get("n", 1),
                seed=self.config.get("seed", 42),
                test_frac=self.config.get("test_frac", 0.2),
                val_frac=self.config.get("val_frac", 0.1),
                cols_to_scale=self.config.get('cols_to_scale', None),
                transform=self.get_transforms(split),
            )
        return self.ds[split]

    def get_dataloader(self, split: str) -> DataLoader:
        """Create and return a DataLoader for the requested split."""
        
        dataset = self.get_dataset(split)


        batch_size = int(self.config.get("batch_size", 2))
        shuffle = bool(self.config.get("shuffle", split == "train"))
        num_workers = int(self.config.get("num_workers", 4))
        pin_memory = bool(self.config.get("pin_memory", True))
        persistent_workers = bool(
            self.config.get("persistent_workers", (num_workers > 0))
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        return loader

    def __call__(self, split: str):
        return self.get_dataloader(split)

    def class_weights(self) -> Optional[Any]:
        """
        Convenience accessor to fetch class weights from an instantiated dataset
        without building a dataloader.
        """
        return self.ds['train'].class_weights

    def class_info(self) -> Dict[str, Any]:
        """
        Convenience accessor to get class mapping info:
          - cls2idx (dict[str,int])
          - idx2cls (dict[int,str])
          - num_classes (int)
        """
        ds = self.ds['train']
        return {
            "cls2idx": ds.cls2idx,
            "idx2cls": ds.idx2cls,
            "num_classes": ds.num_classes,
            "norm_ingr": ds.norm_ingr,
            'class_weights': ds.class_weights,
            'scalers': ds.scalers,
            'img_size': ds.img_size,
        }



class DatasetInfo:
    """
    Stores dataset metadata and can save/load itself to/from JSON.

    Expects `info` dict with keys:
      - cls2idx: Dict[str, int]
      - idx2cls: Dict[int|str, str]   (string keys will be cast back to int)
      - num_classes: int
      - norm_ingr: torch.Tensor | list | None
      - class_weights: torch.Tensor | list | None
      - scalers: Dict[str, StandardScaler | dict] | None
      - img_size: int
    """

    def __init__(self, info: Dict[str, Any]):
        self.cls2idx: Dict[str, int] = info["cls2idx"]
        # ensure idx2cls has int keys
        self.idx2cls: Dict[int, str] = {int(k): v for k, v in info["idx2cls"].items()}
        self.num_classes: int = int(info["num_classes"])

        # new field
        self.img_size: int = int(info.get("img_size", 256))  # default to 256

        # accept tensor or list; store as torch.Tensor or None
        self.norm_ingr: Optional[torch.Tensor] = self._ensure_tensor(info.get("norm_ingr"))
        self.class_weights: Optional[torch.Tensor] = self._ensure_tensor(info.get("class_weights"))

        # scalers can be live objects or already-serialized dicts; normalize to objects here
        raw_scalers = info.get("scalers") or {}
        self.scalers: Dict[str, Any] = {
            k: (self._deserialize_scaler(v) if self._looks_like_serialized_scaler(v) else v)
            for k, v in raw_scalers.items()
        }

    # ---------------- Public API ----------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable dict (tensors -> lists, scalers -> dicts)."""
        return {
            "cls2idx": self.cls2idx,
            "idx2cls": {str(k): v for k, v in self.idx2cls.items()},  # JSON keys must be str
            "num_classes": self.num_classes,
            "img_size": self.img_size,
            "norm_ingr": self.norm_ingr.tolist() if self.norm_ingr is not None else None,
            "class_weights": self.class_weights.tolist() if self.class_weights is not None else None,
            "scalers": {k: self._serialize_scaler(v) for k, v in (self.scalers or {}).items()},
        }

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Saved config → {path}")

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetInfo":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct tensors
        if data.get("norm_ingr") is not None:
            data["norm_ingr"] = torch.tensor(data["norm_ingr"], dtype=torch.float32)
        if data.get("class_weights") is not None:
            data["class_weights"] = torch.tensor(data["class_weights"], dtype=torch.float32)

        # Reconstruct scalers
        ser_scalers = data.get("scalers") or {}
        data["scalers"] = {k: cls._deserialize_scaler(v) for k, v in ser_scalers.items()}

        return cls(data)

    # ---------------- Helpers ----------------
    @staticmethod
    def _ensure_tensor(x: Any) -> Optional[torch.Tensor]:
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.float32)
        if isinstance(x, np.ndarray):  # just in case
            return torch.tensor(x, dtype=torch.float32)
        raise TypeError(f"Unsupported tensor-like type: {type(x)}")

    @staticmethod
    def _looks_like_serialized_scaler(obj: Any) -> bool:
        return isinstance(obj, dict) and obj.get("class") == "StandardScaler"

    @staticmethod
    def _serialize_scaler(scaler: Any) -> Dict[str, Any]:
        """Serialize StandardScaler to a JSON-friendly dict."""
        if isinstance(scaler, StandardScaler):
            return {
                "class": "StandardScaler",
                "mean_": scaler.mean_.tolist(),
                "scale_": scaler.scale_.tolist(),
                "var_": scaler.var_.tolist(),
                "n_samples_seen_": int(scaler.n_samples_seen_),
                "with_mean": bool(getattr(scaler, "with_mean", True)),
                "with_std": bool(getattr(scaler, "with_std", True)),
            }
        raise TypeError(f"Unsupported scaler type for serialization: {type(scaler)}")

    @staticmethod
    def _deserialize_scaler(data: Dict[str, Any]) -> Any:
        """Deserialize a scaler dict back to a StandardScaler."""
        if data.get("class") != "StandardScaler":
            raise ValueError(f"Unsupported scaler class: {data.get('class')}")

        scaler = StandardScaler(
            with_mean=data.get("with_mean", True),
            with_std=data.get("with_std", True),
        )
        scaler.mean_ = np.asarray(data["mean_"], dtype=np.float64)
        scaler.scale_ = np.asarray(data["scale_"], dtype=np.float64)
        scaler.var_ = np.asarray(data["var_"], dtype=np.float64)
        scaler.n_samples_seen_ = int(data["n_samples_seen_"])
        return scaler




    
def compute_class_weights(
    df: pd.DataFrame,
    cls2idx: dict[str, int],
    label_col: str = "label",
    method: str = "balanced",
) -> torch.Tensor:
    """
    Compute per-class weights for multi-label classification.

    Args:
        df: DataFrame with stringified list labels in `label_col`.
        cls2idx: mapping {class_name: index}.
        label_col: column with stringified list of labels.
        method: weighting strategy:
            - "balanced": inverse frequency -> N / (num_classes * count)
            - "inverse": 1 / count
            - "sqrt_inv": 1 / sqrt(count)

    Returns:
        torch.FloatTensor of shape [num_classes] with weights.
    """
    num_classes = len(cls2idx)
    counts = np.zeros(num_classes, dtype=np.int64)

    # --- Count class occurrences across all samples ---
    for labels in df[label_col].dropna().astype(str):
        try:
            parsed = ast.literal_eval(labels)
            for lab in parsed:
                idx = cls2idx.get(lab)
                if idx is not None:
                    counts[idx] += 1
        except Exception:
            continue

    # --- Avoid division by zero ---
    counts = np.maximum(counts, 1)

    if method == "balanced":
        total = counts.sum()
        weights = total / (num_classes * counts)
    elif method == "inverse":
        weights = 1.0 / counts
    elif method == "sqrt_inv":
        weights = 1.0 / np.sqrt(counts)
    else:
        raise ValueError(f"Unknown method: {method}")

    return torch.tensor(weights, dtype=torch.float32)

def build_norm_ingr_table(class_to_idx, norm_ingr_dict):
    """
    class_to_idx: dict like {'eggs': 0, 'chicken': 1, ...}
    portion_independent_dict: dict like {'eggs': {'protein': 13, 'fat': 11, 'carbs': 1},
                             'chicken': {'protein': 27, 'fat': 3, 'carbs': 0}, ...}
    Returns: tensor (C,3) in order [P,F,C]
    """
    C = len(class_to_idx)
    norm_ingr_table = torch.zeros(C, 3, dtype=torch.float32)
    for name, idx in class_to_idx.items():
        m = norm_ingr_dict[name]
        norm_ingr_table[idx, 0] = float(m['protein'])
        norm_ingr_table[idx, 1] = float(m['fat'])
        norm_ingr_table[idx, 2] = float(m['carbs'])
    return norm_ingr_table  # (C,3)

