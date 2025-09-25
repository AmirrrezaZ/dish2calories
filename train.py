from pathlib import Path
import logging
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import MultilabelF1Score


def train_model(data_loader, model, criterion, config):
    """
    Train a model with OneCycleLR and early stopping on validation loss.
    Logs both to console and to logs/logs.txt.
    """

    train_loader = data_loader['train']
    val_loader = data_loader['val']

    # ---- Logging ----
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "logs.txt"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),                # console
            logging.FileHandler(log_file, mode="w") # file
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(
        "Starting training:\n"
        f"  Model: {config.get('model_type')}\n"
        f"  Epochs: {config.get('epochs')}\n"
        f"  Batch size: {config.get('batch_size')}\n"
        f"  Learning rate: {config.get('learning_rate')}\n"
        f"  Device: {config.get('device')}\n"
    )

    device = torch.device(config['device'])
    model.to(device)
    
    
    # Move criterion to device if it has parameters
    # if hasattr(criterion, 'to'):
    #     criterion = criterion.to(device)
    criterion.multitaskloss_instance.to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = Adam(
        list(model.parameters()) +  list(criterion.multitaskloss_instance.parameters()),
        lr=float(config['learning_rate'])
    )
    
    epochs = int(config['epochs'])
    steps_per_epoch = len(train_loader)

    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=float(config.get('max_lr', 3e-3)),
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=float(config.get('warmup_pct', 0.01)),
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # ---- Early stopping ----
    best_val_loss = float('inf')
    patience = int(config.get('patience', 10))
    epochs_without_improvement = 0

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = weights_dir / f"{config['model_type']}.pth"

    grad_clip = float(config.get('grad_clip', 1.0))
    

    train_f1_micro = MultilabelF1Score(num_labels=model.num_classes).to(device)
    val_f1_micro   = MultilabelF1Score(num_labels=model.num_classes).to(device)

    

    # Validation check
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("Train or validation loader is empty")

    try:
        for epoch in range(1, epochs + 1):
            # ---- Train ----
            model.train()
            running_train_loss = 0.0
            train_loss_dict = {}

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")
            train_f1_micro.reset()
            for batch in pbar:
                images = batch['image'].to(device=device, dtype=torch.float32)
                target = {
                    "cls_multi_hot": batch["cls_multi_hot"].to(device=device, dtype=torch.float32),
                    "mass": batch["mass"].to(device=device, dtype=torch.float32),
                }

                optimizer.zero_grad()
                outputs = model(images)
                loss, loss_dict = criterion(target, outputs)
                
                probs = outputs["classification"].sigmoid()
                train_f1_micro.update(probs, target["cls_multi_hot"].int())
                
                

                # Backward pass
                loss.backward()
                
                if grad_clip > 0:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    torch.nn.utils.clip_grad_norm_(
                            list(model.parameters()) + list(criterion.multitaskloss_instance.parameters()),
                            grad_clip
                        )

                    
            
                # Step optimizer & scheduler
                optimizer.step()
                scheduler.step()
                
                running_train_loss += loss.item()
                for k, v in loss_dict.items():
                    val = v.item() if hasattr(v, 'item') else float(v)
                    train_loss_dict[k] = train_loss_dict.get(k, 0.0) + val

                pbar.set_postfix(loss=loss.item())

            avg_train_loss = running_train_loss / max(1, len(train_loader))
            avg_train_loss_dict = {k: v / len(train_loader) for k, v in train_loss_dict.items()}
            avg_train_loss_dict['acc']=train_f1_micro.compute().item()

            # ---- Validate ----
            model.eval()
            val_loss = 0.0
            val_loss_dict = {}
            val_f1_micro.reset()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(device=device, dtype=torch.float32)
                    target = {
                        "cls_multi_hot": batch["cls_multi_hot"].to(device=device, dtype=torch.float32, non_blocking=True),
                        "mass": batch["mass"].to(device=device, dtype=torch.float32),
                    }
                    
                    outputs = model(images)
                    loss, loss_dict = criterion(target, outputs)
                    
                    probs = outputs["classification"].sigmoid()
                    val_f1_micro.update(probs, target["cls_multi_hot"].int())
                    
              

                    val_loss += loss.item()
                    for k, v in loss_dict.items():
                        val = v.item() if hasattr(v, 'item') else float(v)
                        val_loss_dict[k] = val_loss_dict.get(k, 0.0) + val

            avg_val_loss = val_loss / max(1, len(val_loader))
            avg_val_loss_dict = {k: v / len(val_loader) for k, v in val_loss_dict.items()}
            avg_val_loss_dict['acc'] = val_f1_micro.compute().item()

            logger.info(
                f"Epoch {epoch} | "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Train Loss Dict: {avg_train_loss_dict}, "
                f"Val Loss Dict: {avg_val_loss_dict}"
            )

            # ---- Checkpoints & early stopping ----
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
                model.save_model(best_ckpt_path)
                logger.info(f"New best model saved at {best_ckpt_path} (val_loss={best_val_loss:.4f})")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs.")
                break

        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        interrupt_ckpt = weights_dir / f"{config['model_type']}_interrupt.pth"
        model.save_model(interrupt_ckpt)
        logger.info(f"Interrupted checkpoint saved to: {interrupt_ckpt}")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


# ---------------- Main ----------------
if __name__ == "__main__":
    
    from dataset_loaders import Nutrition5kLoaders, DatasetInfo
    from models import CalorieNet
    from loss import NutritionMultiTaskLoss



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
    pos_class_weights = info.get("class_weights", None)
    if pos_class_weights is not None:
        pos_class_weights = torch.tensor(pos_class_weights, dtype=torch.float32, device=config["device"])

    criterion = NutritionMultiTaskLoss(
        regression_type="mae",
        pos_class_weights=pos_class_weights,
        scale_factor=100,
    )

    # ----------------------------
    # Train
    # ----------------------------
    train_model({"train": train_loader, "val": val_loader}, calorie_net, criterion, config)

    