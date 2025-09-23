from dataset_loaders import Nutrition5kLoaders, DatasetInfo
from models import CalorieNet
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
)

import numpy as np
import torch
from typing import Any, Dict, List
from tqdm import tqdm

class NutritionModelEvaluator:
    """Evaluator class for nutrition prediction models."""
   
    def __init__(self,
                 num_labels: int,
                 threshold: float = 0.5,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 metric_keys: List[str] = None):
        """
        Initialize evaluator with metrics.
       
        Args:
            num_labels: Number of classification labels
            threshold: Threshold for classification metrics
            device: Device to run evaluation on
            metric_keys: List of metric keys to initialize ['accuracy', 'precision', 'recall', 'f1', 'mae']
        """
        self.device = device
        self.num_labels = num_labels
        self.threshold = threshold
       
        if metric_keys is None:
            metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'mae']
       
        self.metrics = self.setup_metrics(metric_keys)
   
    @torch.no_grad()
    def predict_nutrition_from_classification(
        self,
        preds: torch.Tensor,  # [N, C] binary {0,1}
        info: "DatasetInfo"   # info.norm_ingr: [C, D] torch Tensor
    ) -> torch.Tensor:
        """Predict nutrition by summing ingredient features of active classes via matrix multiplication.

        Args:
            preds: Binary predictions tensor of shape [N, C] where N is batch size and C is number of classes
            info: DatasetInfo object containing normalized ingredient features (info.norm_ingr: [C, D])

        Returns:
            Tensor of shape [N, D] representing predicted nutrition values
        """
        return preds.float() @ info.norm_ingr.float()
    
    def calculate_calories(self, p: torch.Tensor, f: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Calculate total calories from macronutrients using standard conversion factors.

        Args:
            p: Protein values in grams
            f: Fat values in grams
            c: Carbohydrate values in grams

        Returns:
            Tensor containing total calorie values
        """
        return 4.0 * p + 4.0 * c + 9.0 * f
    
    def setup_metrics(self, metric_keys: List[str]) -> Dict[str, Any]:
        """Initialize and configure evaluation metrics on the specified device.

        Args:
            metric_keys: List of metric names to initialize

        Returns:
            Dictionary mapping metric names to initialized metric objects

        Raises:
            ValueError: If an unknown metric key is provided
        """
        metrics = {}
       
        metric_map = {
            'accuracy': lambda: MultilabelAccuracy(num_labels=self.num_labels, threshold=self.threshold, average='micro').to(self.device),
            'precision': lambda: MultilabelPrecision(num_labels=self.num_labels, threshold=self.threshold, average='micro').to(self.device),
            'recall': lambda: MultilabelRecall(num_labels=self.num_labels, threshold=self.threshold, average='micro').to(self.device),
            'f1': lambda: MultilabelF1Score(num_labels=self.num_labels, threshold=self.threshold, average='micro').to(self.device),
            'mae': lambda: MeanAbsoluteError().to(self.device),
        }
       
        for key in metric_keys:
            if key in metric_map:
                metrics[key] = metric_map[key]()
            else:
                raise ValueError(f"Unknown metric key: {key}. Available keys: {list(metric_map.keys())}")
       
        return metrics
    
    def _process_batch_classification(
        self, logits: torch.Tensor, target_cls: torch.Tensor, cls_threshold: float
    ):
        """Process a batch for classification, updating metrics and collecting per-sample results.

        Args:
            logits: Raw model outputs (logits) for classification
            target_cls: Target classification labels
            cls_threshold: Threshold for converting probabilities to binary predictions

        Returns:
            Tuple containing:
                - preds: Binary predictions
                - probs: Sigmoid probabilities
                - labels_batch: Per-sample predicted labels
                - confs_batch: Per-sample confidences
                - conf_scalar_batch: Per-sample average confidences
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= cls_threshold).int()
        # Update metrics
        self.metrics['accuracy'](probs, target_cls)
        self.metrics['precision'](probs, target_cls)
        self.metrics['recall'](probs, target_cls)
        self.metrics['f1'](probs, target_cls)
        # Collect sample-level labels and confidences
        n = preds.size(0)
        labels_batch, confs_batch, conf_scalar_batch = [[] for _ in range(n)], [[] for _ in range(n)], []
        pred_indices = (preds == 1).nonzero(as_tuple=False)
        if pred_indices.numel() > 0:
            for r, c in pred_indices.tolist():
                labels_batch[r].append(c)
                confs_batch[r].append(float(probs[r, c].item()))
        conf_scalar_batch = [float(np.mean(cs)) if cs else 0.0 for cs in confs_batch]
        return preds, probs, labels_batch, confs_batch, conf_scalar_batch
    
    def _process_batch_mass(self, outputs, batch, info):
        """Process a batch for mass prediction, inverse-transforming and updating MAE metric.

        Args:
            outputs: Model outputs containing predicted mass
            batch: Input batch containing target mass
            info: DatasetInfo object containing scaler for mass

        Returns:
            Tuple containing:
                - pred_mass_g: Predicted mass in grams
                - tgt_mass_g: Target mass in grams
        """
        pred_mass_norm = outputs['mass'].detach().cpu().numpy().reshape(-1, 1)
        tgt_mass_norm = batch['mass'].detach().cpu().numpy().reshape(-1, 1)
        scaler = info.scalers['total_mass']
        pred_mass_g = scaler.inverse_transform(pred_mass_norm).flatten()
        tgt_mass_g = scaler.inverse_transform(tgt_mass_norm).flatten()
        self.metrics['mae'](
            torch.from_numpy(pred_mass_g).to(self.device, dtype=torch.float32),
            torch.from_numpy(tgt_mass_g).to(self.device, dtype=torch.float32),
        )
        return pred_mass_g, tgt_mass_g
    
    def _process_batch_nutrition(self, preds, info, pred_mass_g, batch):
        """Compute total nutrition values and collect per-macronutrient arrays.

        Args:
            preds: Binary classification predictions
            info: DatasetInfo object containing normalized ingredient features
            pred_mass_g: Predicted mass in grams
            batch: Input batch containing target nutrition values

        Returns:
            Dictionary containing predicted and target values for protein, fat, carbs, and calories
        """
        nutrition_100g = self.predict_nutrition_from_classification(preds, info)
        prot_100g, fat_100g, carb_100g = nutrition_100g[:, 0], nutrition_100g[:, 1], nutrition_100g[:, 2]
        cal_100g = self.calculate_calories(prot_100g, fat_100g, carb_100g)
        mass_factor = torch.from_numpy(pred_mass_g).to(self.device, dtype=torch.float32) / 100.0
        prot_total = prot_100g * mass_factor
        fat_total = fat_100g * mass_factor
        carb_total = carb_100g * mass_factor
        cal_total = cal_100g * mass_factor
        # Targets on same device → then move to CPU once
        tgt_prot = batch["protein"].to(self.device, dtype=torch.float32)
        tgt_fat = batch["fat"].to(self.device, dtype=torch.float32)
        tgt_carb = batch["carb"].to(self.device, dtype=torch.float32)
        tgt_cal = batch["calories"].to(self.device, dtype=torch.float32)
        # Return numpy arrays (batch-sized)
        return {
            "protein_pred": prot_total.detach().cpu().numpy(),
            "protein_tgt": tgt_prot.detach().cpu().numpy(),
            "fat_pred": fat_total.detach().cpu().numpy(),
            "fat_tgt": tgt_fat.detach().cpu().numpy(),
            "carbs_pred": carb_total.detach().cpu().numpy(),
            "carbs_tgt": tgt_carb.detach().cpu().numpy(),
            "calories_pred": cal_total.detach().cpu().numpy(),
            "calories_tgt": tgt_cal.detach().cpu().numpy(),
        }
    
    def _finalize_results(
        self, ret_labels, ret_cls_conf, ret_confidence,
        preds_mass_all, targets_mass_all,
        protein, fat, carb, cal, info
    ):
        """Aggregate results, compute MAEs, and build final result dictionary.

        Args:
            ret_labels: List of per-sample predicted labels
            ret_cls_conf: List of per-sample classification confidences
            ret_confidence: List of per-sample average confidences
            preds_mass_all: List of predicted masses
            targets_mass_all: List of target masses
            protein: Tuple of (predicted, target) protein values
            fat: Tuple of (predicted, target) fat values
            carb: Tuple of (predicted, target) carbohydrate values
            cal: Tuple of (predicted, target) calorie values
            info: DatasetInfo object containing class-to-index mapping

        Returns:
            Dictionary containing evaluation metrics and predictions
        """
        results = {
            'accuracy': float(self.metrics['accuracy'].compute().item()),
            'precision': float(self.metrics['precision'].compute().item()),
            'recall': float(self.metrics['recall'].compute().item()),
            'f1_score': float(self.metrics['f1'].compute().item()),
            'mass_mae': float(self.metrics['mae'].compute().item()),
        }
        results.update({
            'protein_mae': float(np.mean(np.abs(protein[0] - protein[1]))),
            'fat_mae': float(np.mean(np.abs(fat[0] - fat[1]))),
            'carbs_mae': float(np.mean(np.abs(carb[0] - carb[1]))),
            'calories_mae': float(np.mean(np.abs(cal[0] - cal[1]))),
        })
        idx2cls = {v: k for k, v in info.cls2idx.items()}
        ret_labels_str = [[idx2cls.get(idx, f"unknown_{idx}") for idx in sample] for sample in ret_labels]
        results['predictions'] = {
            'pred_mass_g': preds_mass_all,
            'target_mass_g': targets_mass_all,
        }
        results['classification'] = {
            'labels': ret_labels,
            'labels_str': ret_labels_str,
            'classification_confidence': ret_cls_conf,
            'confidence': ret_confidence,
        }
        return results
    
    @torch.no_grad()
    def evaluate_model(self, model, test_loader, info, cls_threshold=None) -> Dict[str, Any]:
        """Evaluate the model on a test dataset and return comprehensive results.

        Args:
            model: The model to evaluate
            test_loader: DataLoader for the test dataset
            info: DatasetInfo object containing metadata and scalers
            cls_threshold: Optional classification threshold (defaults to self.threshold)

        Returns:
            Dictionary containing evaluation metrics, predictions, and classification results
        """
        if cls_threshold is None:
            cls_threshold = self.threshold
        model.eval()
        info.norm_ingr = info.norm_ingr.to(self.device)
        # Accumulators for MAE computation
        all_pred_protein, all_tgt_protein = [], []
        all_pred_fat, all_tgt_fat = [], []
        all_pred_carbs, all_tgt_carbs = [], []
        all_pred_cal, all_tgt_cal = [], []
        # Accumulators for output (per-sample)
        protein_pred_all, protein_tgt_all = [], []
        fat_pred_all, fat_tgt_all = [], []
        carbs_pred_all, carbs_tgt_all = [], []
        calories_pred_all, calories_tgt_all = [], []
        ret_labels, ret_cls_conf, ret_confidence = [], [], []
        preds_mass_all, targets_mass_all = [], []
        for batch in tqdm(test_loader, total=len(test_loader), desc="Evaluating"):
            images = batch["image"].to(self.device, dtype=torch.float32)
            outputs = model(images)
            preds, probs, labels_batch, confs_batch, conf_scalar_batch = \
                self._process_batch_classification(
                    outputs["classification"], batch["cls_multi_hot"].to(self.device).int(), cls_threshold
                )
            ret_labels.extend(labels_batch)
            ret_cls_conf.extend(confs_batch)
            ret_confidence.extend(conf_scalar_batch)
            pred_mass_g, tgt_mass_g = self._process_batch_mass(outputs, batch, info)
            preds_mass_all.append(pred_mass_g)
            targets_mass_all.append(tgt_mass_g)
            nut = self._process_batch_nutrition(preds, info, pred_mass_g, batch)
            # Store for output
            protein_pred_all.append(nut["protein_pred"]); protein_tgt_all.append(nut["protein_tgt"])
            fat_pred_all.append(nut["fat_pred"]); fat_tgt_all.append(nut["fat_tgt"])
            carbs_pred_all.append(nut["carbs_pred"]); carbs_tgt_all.append(nut["carbs_tgt"])
            calories_pred_all.append(nut["calories_pred"]); calories_tgt_all.append(nut["calories_tgt"])
            # Store for MAE aggregation (reuse same arrays)
            all_pred_protein.append(nut["protein_pred"]); all_tgt_protein.append(nut["protein_tgt"])
            all_pred_fat.append(nut["fat_pred"]); all_tgt_fat.append(nut["fat_tgt"])
            all_pred_carbs.append(nut["carbs_pred"]); all_tgt_carbs.append(nut["carbs_tgt"])
            all_pred_cal.append(nut["calories_pred"]); all_tgt_cal.append(nut["calories_tgt"])
        # Scalar metrics
        results: Dict[str, Any] = {
            "accuracy": float(self.metrics["accuracy"].compute().item()),
            "precision": float(self.metrics["precision"].compute().item()),
            "recall": float(self.metrics["recall"].compute().item()),
            "f1_score": float(self.metrics["f1"].compute().item()),
            "mass_mae": float(self.metrics["mae"].compute().item()),
        }
        # MAEs (concat once)
        ap_prot = np.concatenate(all_pred_protein); at_prot = np.concatenate(all_tgt_protein)
        ap_fat = np.concatenate(all_pred_fat); at_fat = np.concatenate(all_tgt_fat)
        ap_carb = np.concatenate(all_pred_carbs); at_carb = np.concatenate(all_tgt_carbs)
        ap_cal = np.concatenate(all_pred_cal); at_cal = np.concatenate(all_tgt_cal)
        results.update({
            "protein_mae": float(np.mean(np.abs(ap_prot - at_prot))),
            "fat_mae": float(np.mean(np.abs(ap_fat - at_fat))),
            "carbs_mae": float(np.mean(np.abs(ap_carb - at_carb))),
            "calories_mae": float(np.mean(np.abs(ap_cal - at_cal))),
        })
        # Labels as strings
        idx2cls = {v: k for k, v in info.cls2idx.items()}
        ret_labels_str = [[idx2cls.get(i, f"unknown_{i}") for i in lst] for lst in ret_labels]
        # Pack outputs (including macros)
        results["predictions"] = {
            "pred_mass_g": preds_mass_all,  # List of np.ndarrays (per-batch)
            "target_mass_g": targets_mass_all,
            "protein": {
                "pred": protein_pred_all,  # List of np.ndarrays
                "tgt": protein_tgt_all,
            },
            "fat": {
                "pred": fat_pred_all,
                "tgt": fat_tgt_all,
            },
            "carbs": {
                "pred": carbs_pred_all,
                "tgt": carbs_tgt_all,
            },
            "calories": {
                "pred": calories_pred_all,
                "tgt": calories_tgt_all,
            },
        }
        results["classification"] = {
            "labels": ret_labels,
            "labels_str": ret_labels_str,
            "classification_confidence": ret_cls_conf,
            "confidence": ret_confidence,
        }
        return results

# ----------------------------
# Config
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    # Required (dataset side)
    "base_dir": "dataset",        # dataset root with images + metadata
    "img_size": 256,              
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
    "pin_memory": torch.cuda.is_available(),
    # enable persistent workers only if workers > 0
    "persistent_workers": True,
    "device": device,
}

# Adjust worker flags safely
if config["num_workers"] <= 0:
    config["persistent_workers"] = False

# ----------------------------
# Data
# ----------------------------
info = DatasetInfo.from_json("configs/dataset_info.json")

config['cols_to_scale'] = list(info.scalers.keys())

ds = Nutrition5kLoaders(config)
test_loader = ds("test")

# ----------------------------
# Model
# ----------------------------
calorie_net, _ = CalorieNet.from_checkpoint(path=f"weights/{config['model_type']}.pth", device=device)

# ----------------------------
# Evaluation
# ----------------------------
# Create evaluator instance with metrics setup
num_labels = info.num_classes 
threshold = 0.5
metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'mae']

evaluator = NutritionModelEvaluator(
    num_labels=num_labels,
    threshold=threshold,
    device=device,
    metric_keys=metric_keys
)

# Run evaluation
results = evaluator.evaluate_model(
    model=calorie_net,
    test_loader=test_loader,
    info=info,
    cls_threshold=threshold
)

# Print results
print("Evaluation Results:")
for key, value in results.items():
    if key not in ['predictions', 'classification']:
        print(f"{key}: {value}")
        
        
