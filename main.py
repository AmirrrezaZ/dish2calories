from dataset_loaders import Nutrition5kLoaders, DatasetInfo
from models import CalorieNet
from loss import NutritionMultiTaskLoss
from train import train_model

import torch


# ----------------------------
# Config
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
config = {
    # Required (dataset side)
    "base_dir": "dataset",        # dataset root with images + metadata
    "img_size": 256,              # resize target for all images
    "split": "train",             # one of ["train", "val", "test"]
    "model_type": "efficientnet_b0",
    
    "cols_to_scale": ['total_mass'],

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

    # Model/training
    "pretrained": True,           # or pass a torchvision weights enum in  CalorieNet
    "epochs": 50,
    "learning_rate": 1e-3,
    "max_lr": 3e-3,
    "warmup_pct": 0.01,
    "patience": 10,
    "grad_clip": 1.0,
    "device": device,

    # Optional model head tweaks
    "hidden_dim": 64,
    "dropout_rate": 0.3,
}

# Adjust worker flags safely
if config["num_workers"] <= 0:
    config["persistent_workers"] = False


# ----------------------------
# Data
# ----------------------------
ds = Nutrition5kLoaders(config)
train_loader = ds("train")
val_loader = ds("val")
info = ds.class_info()  # expects {'num_classes': int, 'class_weights': array-like or None, ...}

cfg = DatasetInfo(info)
cfg.save_json("configs/dataset_info.json")



# ----------------------------
# Model
# ----------------------------
calorie_net = CalorieNet(
    model_name=config["model_type"],
    num_classes=info["num_classes"],
    pretrained=config.get("pretrained", True),
    device=config["device"],
    hidden_dim=config.get("hidden_dim", 64),
    dropout_rate=config.get("dropout_rate", 0.3),
)



# ----------------------------
# Loss
# ----------------------------
class_weights = info.get("class_weights", None)
if class_weights is not None:
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=config["device"])

criterion = NutritionMultiTaskLoss(
    label_smoothing=0.05,
    regression_type="mae",
    class_weights=class_weights,
)

# ----------------------------
# Train
# ----------------------------
train_model({"train": train_loader, "val": val_loader}, calorie_net, criterion, config)
