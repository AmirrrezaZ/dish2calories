import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class NutritionLoss(nn.Module):
    """
    Multi-task loss for nutrition prediction.

    Combines:
      1) Classification (multi-label): BCEWithLogitsLoss
      2) Mass regression: MAE (L1) or MSE (L2)

    Args
    ----
    lambda_cls: float = 1.0
        Weight for classification loss.
    lambda_mass: float = 1.0
        Weight for mass regression loss.
    regression_type: {"mae", "mse"} = "mae"
        Which regression loss to use.
    pos_class_weights: Optional[torch.Tensor] = None
        Per-class positive weights for BCE (shape [C]), passed as `pos_weight`.
        Use this for class imbalance (higher value => penalize FN more).

    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_mass: float = 1.0,
        regression_type: str = "mae",
        pos_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # classification loss (multi-label)
        # NOTE: use pos_weight for per-class positive weighting
        self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_class_weights)

        # regression loss
        regression_type = regression_type.lower()
        if regression_type == "mae":
            self.reg_loss = nn.L1Loss()
        elif regression_type == "mse":
            self.reg_loss = nn.MSELoss()
        else:
            raise ValueError(f"Invalid regression_type {regression_type!r}. Use 'mae' or 'mse'.")

        self.lambda_cls = float(lambda_cls)
        self.lambda_mass = float(lambda_mass)


    @staticmethod
    def _to_multi_hot(
        target: Dict[str, torch.Tensor],
        num_classes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Accept any of:
          - target['cls_multi_hot'] (preferred): [B, C] with 0/1
          - target['cls_one_hot']            : [B, C] with 0/1 (alias)
          - target['class']                  : [B] indices (single-label); will be converted to one-hot
        Returns float tensor [B, C].
        """
        if "cls_multi_hot" in target:
            y = target["cls_multi_hot"].to(device=device, dtype=dtype)
            if y.shape[-1] != num_classes:
                raise ValueError("cls_multi_hot has wrong number of classes.")
            return y

        if "cls_one_hot" in target:
            y = target["cls_one_hot"].to(device=device, dtype=dtype)
            if y.shape[-1] != num_classes:
                raise ValueError("cls_one_hot has wrong number of classes.")
            return y

        if "class" in target:
            # single-label indices -> one-hot
            idx = target["class"].to(device=device, dtype=torch.long)
            if idx.ndim != 1:
                raise ValueError("target['class'] should be shape [B] with class indices.")
            B = idx.shape[0]
            y = torch.zeros(B, num_classes, device=device, dtype=dtype)
            y.scatter_(1, idx.unsqueeze(1), 1.0)
            return y

        raise KeyError("Provide 'cls_multi_hot' (preferred), 'cls_one_hot', or 'class' in target.")

    def forward(
        self,
        target: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # ----- unpack predictions
        if "classification" not in output:
            raise KeyError("Output dict must contain key 'classification' with logits [B, C].")
        logits = output["classification"]                       # [B, C]
        if logits.ndim != 2:
            raise ValueError("output['classification'] must be [B, C].")

        pred_mass = output.get("mass")                          # [B] or [B,1]
        if pred_mass is None:
            raise KeyError("Output dict must contain key 'mass'.")
        pred_mass = pred_mass.squeeze(-1)                       # -> [B]

        # ----- targets
        C = logits.shape[1]
        device = logits.device
        target_class = self._to_multi_hot(target, C, device, logits.dtype)  # [B, C] float in {0,1}

        target_mass = target.get("mass")
        if target_mass is None:
            raise KeyError("Target dict must contain key 'mass'.")
        target_mass = target_mass.squeeze(-1).to(device=device, dtype=pred_mass.dtype)  # [B]

        # ----- 1) classification loss (multi-label BCE with logits)
        loss_cls = self.ce(logits, target_class)

        # ----- 2) regression (MAE or MSE)
        loss_mass = self.reg_loss(pred_mass, target_mass)

        # ----- weighted sum
        total = self.lambda_cls * loss_cls + self.lambda_mass * loss_mass


        loss_dict = {
            "total": total,
            "cls": self.lambda_cls,
            "mass": self.lambda_mass,
        }

        return total, loss_dict



class MultiTaskLoss(torch.nn.Module):
    '''https://arxiv.org/abs/1705.07115'''
    def __init__(self, is_regression, reduction='none'):
        super(MultiTaskLoss, self).__init__()
        self.is_regression = is_regression
        self.n_tasks = len(is_regression)
        self.log_vars = torch.nn.Parameter(torch.zeros(self.n_tasks))
        self.reduction = reduction

    def forward(self, losses):
        dtype = losses.dtype
        device = losses.device
        stds = (torch.exp(self.log_vars)**(1/2)).to(device).to(dtype)
        self.is_regression = self.is_regression.to(device).to(dtype)
        coeffs = 1 / ( (self.is_regression+1)*(stds**2) )
        multi_task_losses = coeffs*losses + torch.log(stds)

        if self.reduction == 'sum':
            multi_task_losses = multi_task_losses.sum()
        if self.reduction == 'mean':
            multi_task_losses = multi_task_losses.mean()

        return multi_task_losses


class NutritionMultiTaskLoss:
    """
    Multi-task loss for nutrition prediction.

    This loss combines:
      1) **Classification loss**: CrossEntropyLoss with optional class weights
         and label smoothing.
      2) **Mass regression loss**: Either MAE (L1) or MSE (L2), selectable.

    Parameters
    ----------
    regression_type : {"mae", "mse"}, default="mae"
        Which regression loss to use for mass prediction.


    pos_class_weights : torch.Tensor or None, optional
        Per-class weights for classification loss. Shape [num_classes].
    """

    def __init__(
        self,
        regression_type: str = "mae",
        pos_class_weights: Optional[torch.Tensor] = None,
        scale_factor: int = 100,
    ):
        self.criterion = NutritionLoss(
            regression_type=regression_type,
            pos_class_weights=pos_class_weights,
        )
        
        self.multitaskloss_instance = MultiTaskLoss(
            is_regression=torch.Tensor([False, True]),
            reduction='sum'
        )
        self.scale_factor = scale_factor

    def __call__(
        self, 
        target: Dict[str, torch.Tensor],
        output: Dict[str, torch.Tensor]
    ):
        _, loss_dict = self.criterion(target, output)
        
        L_cls = loss_dict["cls"] * self.scale_factor
        L_mass = loss_dict["mass"] * self.scale_factor
   
        losses = torch.stack([L_cls, L_mass])
        mt_loss = self.multitaskloss_instance(losses)
        
        loss_dict['total'] = mt_loss
        
        return mt_loss, loss_dict

        
        

    
    