import os
import warnings
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.vision_transformer import VisionTransformer

warnings.filterwarnings("ignore")


class RegressionHead(nn.Module):
    def __init__(self, input_features: int, output_dim: int = 1, hidden_dim: int = 64, dropout_rate: float = 0.5):
        super().__init__()
        self.dense = nn.Linear(input_features, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.dense(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class NutritionModel(nn.Module):
    """
    Multi-task model: classification + regression.
    - classification_head: dish categories
    - regression_head: predicts total mass
    """
    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        feature_layer_index: int = -2,  # kept for API compatibility; unused
        hidden_dim: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        # Build feature extractor
        self.backbone_features = self._create_feature_extractor(backbone, feature_layer_index)

        # Infer feature dimension w/o dummy forward
        feature_dim = self._infer_feature_dim(backbone)
        self.feature_dim = feature_dim

        # Heads
        self.classification_head = nn.Linear(feature_dim, num_classes)
        self.mass_head = RegressionHead(feature_dim, output_dim=1, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    @staticmethod
    def _infer_feature_dim(backbone: nn.Module) -> int:
        # ViT in torchvision exposes hidden_dim and classifier as .heads.head
        if isinstance(backbone, VisionTransformer):
            if hasattr(backbone, "hidden_dim"):
                return int(backbone.hidden_dim)
            # Fallback: pull from the final linear in heads
            head = getattr(backbone, "heads", None)
            if head is not None and hasattr(head, "head") and isinstance(head.head, nn.Linear):
                return int(head.head.in_features)

        # ResNet family
        if hasattr(backbone, "fc") and isinstance(backbone.fc, nn.Linear):
            return int(backbone.fc.in_features)

        # DenseNet / ConvNeXt / EfficientNet / MobileNet / VGG expose .classifier
        if hasattr(backbone, "classifier"):
            cls = backbone.classifier
            if isinstance(cls, nn.Linear):
                return int(cls.in_features)
            if isinstance(cls, nn.Sequential):
                # find last Linear and use its in_features
                for m in reversed(cls):
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)

        # Generic fallback: search for final Linear on the module
        last_linear = None
        for m in backbone.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            return int(last_linear.in_features)

        raise ValueError("Could not infer feature dimension from backbone.")

    def _create_feature_extractor(self, backbone: nn.Module, layer_index: int) -> nn.Module:
        # ViT: take encoder output after final LN (pre-head)
        if isinstance(backbone, VisionTransformer):
            return create_feature_extractor(backbone, return_nodes={'encoder.ln': 'feat'})

        # CNN-like backbones
        if hasattr(backbone, 'fc'):  # ResNet-style
            modules = list(backbone.children())[:-1]
            feature_extractor = nn.Sequential(*modules)
        elif hasattr(backbone, 'classifier'):  # VGG/EfficientNet/MobileNet/ConvNeXt/DenseNet
            modules = [m for n, m in backbone.named_children() if n != 'classifier']
            feature_extractor = nn.Sequential(*modules)
        else:
            modules = list(backbone.children())[:-1]
            feature_extractor = nn.Sequential(*modules)

        # Pool to (1,1) then flatten so the extractor output is (N, C)
        feature_extractor.add_module('adaptive_pool', nn.AdaptiveAvgPool2d((1, 1)))
        feature_extractor.add_module('flatten', nn.Flatten())
        return feature_extractor

    def forward(self, x) -> Dict[str, torch.Tensor]:
        feats = self.backbone_features(x)
        feat = feats['feat'] if isinstance(feats, dict) else feats

        # Handle (N, D) vs (N, L, D) for ViT
        if feat.dim() == 3:
            features = feat[:, 0, :]  # class token
        else:
            features = feat

        pred_class = self.classification_head(features)
        pred_mass = self.mass_head(features)  # (N, 1)

        return {'classification': pred_class, 'mass': pred_mass}


class CalorieNet(nn.Module):
    """
    Wrapper that builds a torchvision backbone and the multi-task NutritionModel.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: Optional[Union[bool, str]] = None,  # allow bool or enum string
        device: Optional[Union[str, torch.device]] = None,
        hidden_dim: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()  
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = str(model_name).lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.supported_models = None

        self.backbone = self._create_backbone()
        self.model = NutritionModel(
            backbone=self.backbone,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        ).to(self.device)

    def __call__(self, x):
        return self.forward(x)

    def _create_backbone(self) -> nn.Module:
        model_dict = {
            "alexnet": models.alexnet,
            "convnext_tiny": models.convnext_tiny,
            "convnext_small": models.convnext_small,
            "convnext_base": models.convnext_base,
            "convnext_large": models.convnext_large,
            "densenet121": models.densenet121,
            "densenet169": models.densenet169,
            "densenet201": models.densenet201,
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
            "efficientnet_b5": models.efficientnet_b5,
            "efficientnet_b6": models.efficientnet_b6,
            "efficientnet_b7": models.efficientnet_b7,
            "efficientnet_v2_s": models.efficientnet_v2_s,
            "efficientnet_v2_m": models.efficientnet_v2_m,
            "efficientnet_v2_l": models.efficientnet_v2_l,
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet152,
            "resnext50_32x4d": models.resnext50_32x4d,
            "resnext101_32x8d": models.resnext101_32x8d,
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19,
            "vit_b_16": models.vit_b_16,
            "vit_b_32": models.vit_b_32,
            "vit_l_16": models.vit_l_16,
            "mobilenet_v2": models.mobilenet_v2,
            "mobilenet_v3_small": models.mobilenet_v3_small,
            "mobilenet_v3_large": models.mobilenet_v3_large,
        }
        self.supported_models = list(model_dict.keys())
        if self.model_name not in model_dict:
            raise ValueError(f"Model '{self.model_name}' is not supported.")

        creator = model_dict[self.model_name]
        # In newer torchvision, pass enums like weights=models.ResNet50_Weights.IMAGENET1K_V1
        backbone = creator(weights=self.pretrained)
        return backbone

    def forward(self, x):
        return self.model(x)

    def save_model(self, path: str, save_optimizer: bool = False, optimizer=None, epoch: int = None, loss: float = None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        cfg = {
            "layout": "mass",
            "hidden_dim": self.model.mass_head.dense.out_features,
            "dropout_rate": self.model.mass_head.dropout.p,
            "mass_out": getattr(self.model.mass_head.output, "out_features", None),
        }

        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'model_config': cfg
        }
        if save_optimizer and optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            save_dict['epoch'] = epoch
        if loss is not None:
            save_dict['loss'] = loss

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, optimizer=None, device: Optional[Union[str, torch.device]] = None):
        device = device or self.device
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        metadata = {}
        if 'epoch' in checkpoint:
            metadata['epoch'] = checkpoint['epoch']
        if 'loss' in checkpoint:
            metadata['loss'] = checkpoint['loss']
        if 'model_config' in checkpoint:
            metadata['model_config'] = checkpoint['model_config']
        print(f"Model loaded from {path}")
        return metadata

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.model.named_parameters(prefix, recurse)
    
    @classmethod
    def from_checkpoint(cls, path: str, device: Optional[Union[str, torch.device]] = None, optimizer=None):
        """
        Rebuild a CalorieNet instance entirely from a saved checkpoint.

        Args:
            path: Path to checkpoint (.pth file).
            device: torch.device or str (defaults to cuda if available).
            optimizer: Optional optimizer to restore.

        Returns:
            model: CalorieNet instance with weights loaded.
            metadata: dict with optional 'epoch', 'loss', 'model_config'.
        """
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(path, map_location=device)

        # required fields
        model_name = checkpoint["model_name"]
        num_classes = checkpoint["num_classes"]

        # recover model_config (optional extras)
        cfg = checkpoint.get("model_config", {})
        hidden_dim = cfg.get("hidden_dim", 64)
        dropout_rate = cfg.get("dropout_rate", 0.5)

        # rebuild model with same params
        model = cls(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=None,  # don't load torchvision weights, we have our own
            device=device,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )

        # load weights
        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.model.to(device)

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # collect metadata
        metadata = {}
        for k in ["epoch", "loss", "model_config"]:
            if k in checkpoint:
                metadata[k] = checkpoint[k]

        print(f"[INFO] Loaded {model_name} checkpoint from {path}")
        return model, metadata








if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = 'efficientnet_b1'
    num_classes = 75
    
    calorie_net = CalorieNet(model_name=model_name, num_classes=num_classes, device=device)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 512, 512).to(device)

    calorie_net.eval()
    with torch.no_grad():
        outputs = calorie_net(dummy_input)

    print("✓ Forward pass successful")
    print("Output shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {list(value.shape)}")
    
    # Test save/load
    test_path = f"test_{model_name}.pth"
    calorie_net.save_model(test_path)
    print("✓ Model saved successfully")
    
    # Load model
    new_model = CalorieNet(model_name=model_name, num_classes=num_classes, device=device)
    metadata = new_model.load_model(test_path)
    print("✓ Model loaded successfully")
    print("Metadata:", metadata)
    
    # Clean up
    if os.path.exists(test_path):
        os.remove(test_path)
        
    

