from dataset_loaders import Nutrition5kLoaders, DatasetInfo
from models import CalorieNet
import torch





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

config['cols_to_scale'] =list(info.scalers.keys())

ds = Nutrition5kLoaders(config)
test_loader = ds("test")


# ----------------------------
# Model
# ----------------------------
calorie_net,_ = CalorieNet.from_checkpoint(path=f"weights/{config['model_type']}.pth", device=device)



    

