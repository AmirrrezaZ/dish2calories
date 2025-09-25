from dataset_loaders import Nutrition5kLoaders, DatasetInfo
from models import CalorieNet
from predict import CalorieNetPredictor


import numpy as np
import pandas as pd
from typing import Dict, Any, Literal, Tuple
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

class NutritionEvaluator:
    """
    Evaluate nutrition prediction results against ground-truth targets.

    Metrics:
      - Direct MAE (portion-dependent): total_protein, total_fat, total_carbs, total_calories, total_mass
      - Portion-independent MAE (scaled-to-target-mass): total_protein, total_fat, total_carbs, total_calories
      - Multi-label classification: accuracy, precision, recall, f1 (micro average)
    """

    def __init__(self, dedup: Literal["first", "aggregate"] = "first"):
        self.dedup = dedup

    @staticmethod
    def _build_preds_df(results: Dict[str, Any]) -> pd.DataFrame:
        per_gram = np.asarray(results["per_gram_macros"])  # [N, D] with [P,F,C,...]
        preds_df = pd.DataFrame({
            "img_path": results["img_path"],
            "pred_total_protein": results["total_protein"],
            "pred_total_fat": results["total_fat"],
            "pred_total_carbs": results["total_carbs"],
            "pred_total_calories": results["total_calories"],
            "pred_total_mass": results["total_mass"],
            "pred_pergram_protein": per_gram[:, 0],
            "pred_pergram_fat": per_gram[:, 1],
            "pred_pergram_carbs": per_gram[:, 2],
        })
        preds_df["pred_pergram_calories"] = (
            4.0 * preds_df["pred_pergram_protein"]
            + 9.0 * preds_df["pred_pergram_fat"]
            + 4.0 * preds_df["pred_pergram_carbs"]
        )
        return preds_df

    @staticmethod
    def _aggregate_preds(preds_df: pd.DataFrame, results: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, list]:
        cont_cols = [c for c in preds_df.columns if c != "img_path"]
        agg_df = preds_df.groupby("img_path", as_index=False)[cont_cols].mean()

        from collections import defaultdict
        preds_bin = np.asarray(results["preds_bin"])  # [N, C]
        bin_map = defaultdict(list)
        for pth, row_bin in zip(results["img_path"], preds_bin):
            bin_map[pth].append(row_bin)

        img_order = list(agg_df["img_path"])
        agg_preds_bin = np.stack(
            [(np.sum(bin_map[p], axis=0) > 0).astype(int) for p in img_order],
            axis=0
        )
        return agg_df, agg_preds_bin, img_order

    @staticmethod
    def _keep_first_preds(preds_df: pd.DataFrame, results: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray, list]:
        preds_df_uniq = preds_df.drop_duplicates(subset=["img_path"], keep="first").copy()

        pred_bin_map = {}
        for pth, row_bin in zip(results["img_path"], np.asarray(results["preds_bin"])):
            if pth not in pred_bin_map:
                pred_bin_map[pth] = row_bin

        img_order = list(preds_df_uniq["img_path"])
        y_pred_bin = np.stack([pred_bin_map[p] for p in img_order], axis=0)
        return preds_df_uniq, y_pred_bin, img_order

    @staticmethod
    def _safe_to_np(x) -> np.ndarray:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def evaluate(self, results: Dict[str, Any], targets_df: pd.DataFrame) -> Dict[str, Any]:
        preds_df = self._build_preds_df(results)

        if self.dedup == "aggregate":
            preds_df_use, y_pred_bin, img_order = self._aggregate_preds(preds_df, results)
        else:
            preds_df_use, y_pred_bin, img_order = self._keep_first_preds(preds_df, results)

        targets_uniq = targets_df.drop_duplicates(subset=["img_path"], keep="first").copy()
        df = pd.merge(preds_df_use, targets_uniq, on="img_path", suffixes=("_pred", "_true"))

        path_to_idx = {p: i for i, p in enumerate(img_order)}
        keep_idx = [path_to_idx[p] for p in df["img_path"].tolist() if p in path_to_idx]
        y_pred_bin = y_pred_bin[keep_idx]

        y_true_bin = np.stack([self._safe_to_np(v) for v in df["cls_multi_hot"].values], axis=0)
        assert y_true_bin.shape == y_pred_bin.shape, f"Classification shape mismatch: {y_true_bin.shape} vs {y_pred_bin.shape}"

        # Direct MAE (portion-dependent)
        mae_direct = {}
        for pred_col, true_col in [
            ("pred_total_protein",  "total_protein"),
            ("pred_total_fat",      "total_fat"),
            ("pred_total_carbs",    "total_carb"),
            ("pred_total_calories", "total_calories"),
            ("pred_total_mass",     "total_mass"),
        ]:
            y_true = df[true_col].to_numpy()
            y_pred = df[pred_col].to_numpy()
            mae_direct[f"MAE_{true_col}"] = mean_absolute_error(y_true, y_pred)

        # Portion-independent MAE (scaled to target mass)
        mass_true = df["total_mass"].to_numpy()
        pred_scaled_protein  = df["pred_pergram_protein"].to_numpy()  * mass_true
        pred_scaled_fat      = df["pred_pergram_fat"].to_numpy()      * mass_true
        pred_scaled_carbs    = df["pred_pergram_carbs"].to_numpy()    * mass_true
        pred_scaled_calories = df["pred_pergram_calories"].to_numpy() * mass_true

        mae_portion_independent = {
            "MAE_total_protein":  mean_absolute_error(df["total_protein"].to_numpy(),  pred_scaled_protein),
            "MAE_total_fat":      mean_absolute_error(df["total_fat"].to_numpy(),      pred_scaled_fat),
            "MAE_total_carbs":    mean_absolute_error(df["total_carb"].to_numpy(),     pred_scaled_carbs),
            "MAE_total_calories": mean_absolute_error(df["total_calories"].to_numpy(), pred_scaled_calories),
        }

        # Multi-label classification (micro average)
        clf = {
            "accuracy":  accuracy_score(y_true_bin, y_pred_bin),
            "precision": precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "recall":    recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
            "f1":        f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        }

        return {
            "mae_direct": mae_direct,
            "mae_portion_independent": mae_portion_independent,
            "classification": clf,
            "num_samples": int(len(df)),
            "merged_df": df,
        }




# ---------------- Main ----------------
if __name__ == "__main__":
    
    # ----------------------------
    # Config
    # ----------------------------
    config = {
        # Required (dataset side)
        "base_dir": "dataset",        # dataset root with images + metadata
        "split": "test",              # one of ["train", "val", "test"]
        "model_type": "efficientnet_b0",
        
        # Optional
        "n": 1,
        "seed": 42,
        "test_frac": 0.2,
        "val_frac": 0.1,
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 8,
        # enable pin_memory only on CUDA
        "pin_memory": True,
        # enable persistent workers only if workers > 0
        "persistent_workers": True,
        "device": 'cuda',
    
    }
    
    # Adjust worker flags safely
    if config["num_workers"] <= 0:
        config["persistent_workers"] = False
    
    # ----------------------------
    # Data
    # ----------------------------
    info = DatasetInfo.from_json("configs/dataset_info.json")
    config['cols_to_scale'] = list(info.scalers.keys())
    config['img_size'] = info.img_size
    
    ds = Nutrition5kLoaders(config)
    test_loader = ds("test")
    
    # ----------------------------
    # Model
    # ----------------------------
    calorie_net, _ = CalorieNet.from_checkpoint(path=f"weights/{config['model_type']}.pth")
    
    # ----------------------------
    # Evaluation
    # ----------------------------
    predictor = CalorieNetPredictor(
        model = calorie_net,
        norm_ingr = info.norm_ingr,
        scaler = info.scalers['total_mass'],                  
        idx2cls = info.idx2cls,          
        img_size = info.img_size,
        cls_threshold = 0.5,
        temperature= 0.5,
        mode= "probs",
        device = config['device'],
    )
       
    results = predictor.predict_dataset(test_loader)   
    
    targets_df = test_loader.dataset.df_unscaled
    
      
    evaluator = NutritionEvaluator()   # or dedup="aggregate"
    report = evaluator.evaluate(results, targets_df)
    
    print("=== Direct MAE ===")
    for k, v in report["mae_direct"].items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Portion-independent MAE ===")
    for k, v in report["mae_portion_independent"].items():
        print(f"{k}: {v:.4f}")
    
    print("\n=== Multi-label ===")
    for k, v in report["classification"].items():
        print(f"{k}: {v:.4f}")
    
    print("\nSamples used:", report["num_samples"])
    
