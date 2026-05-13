from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import os
from pathlib import Path
from typing import Dict
import importlib

import gdown
import torch
import torch.nn.functional as F

from src.app.heatmap import gradcam_visualization
import src.models.cnn_trainer as cnn_model_mod
import src.models.resnet_trainer as resnet_mod
import src.core.tools as tools_mod
from src.core.paths import PROJECT_ROOT

CNNTrainer = cnn_model_mod.CNNTrainer
ResNetTrainer = resnet_mod.ResNetTrainer
new_image = tools_mod.new_image

@dataclass
class PredictionResult:
    label: str
    confidence: float
    cat_prob: float
    dog_prob: float

GDRIVE_CHECKPOINT_FOLDER_URL = "https://drive.google.com/drive/folders/1yDvIzrKw8VuGUsEBLb_SPzfyDdnQmL54"


def _download_checkpoints_from_drive() -> None:
    output_dir = PROJECT_ROOT
    try:
        # Tải toàn bộ thư mục checkpoint về thư mục dự án
        gdown.download_folder(
            url=GDRIVE_CHECKPOINT_FOLDER_URL,
            output=str(output_dir),
            quiet=True,
            use_cookies=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Không tải được checkpoints từ Google Drive: {exc}"
        ) from exc


def _checkpoint_exists(model_key: str, params: dict) -> bool:
    for candidate in _checkpoint_candidates(model_key, params):
        if os.path.exists(candidate):
            return True
    return False


def _ensure_checkpoint_available(model_key: str, params: dict) -> None:
    if _checkpoint_exists(model_key, params):
        return

    # Nếu chưa có checkpoint trên local, tải từ Google Drive.
    _download_checkpoints_from_drive()

    if not _checkpoint_exists(model_key, params):
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint cho model {model_key} sau khi tải từ Google Drive."
        )


class BaseModel(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: str) -> PredictionResult:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, val_loader):
        raise NotImplementedError

    @abstractmethod
    def torch_model(self) -> torch.nn.Module:
        raise NotImplementedError

class CNNModel(BaseModel):
    def __init__(self, params: dict, use_se: bool = False):
        p = copy.deepcopy(params)
        p["architecture"] = "se" if use_se else "cnn"
        super().__init__(p)
        self.use_se = use_se
        self.trainer = CNNTrainer(self.params, use_se=self.use_se)

    def load_model(self):
        self.trainer.load_model(load_optimizer=False)
        self.trainer.model.eval()

    def predict(self, image_path: str) -> PredictionResult:
        self.trainer.model.eval()
        size = int(self.params.get("size", 224))
        
        x = torch.tensor(new_image(image_path, size), dtype=torch.float32).view(-1, 3, size, size)
        x = x.to(self.trainer.device)
        
        with torch.no_grad():
            logits = self.trainer.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            
        cat_prob = float(probs[0].item())
        dog_prob = float(probs[1].item())
        label = "cat" if cat_prob >= dog_prob else "dog"
        confidence = max(cat_prob, dog_prob)
        return PredictionResult(label=label, confidence=confidence, cat_prob=cat_prob, dog_prob=dog_prob)

    def evaluate(self, val_loader):
        return self.trainer.evaluate(val_loader)

    def torch_model(self) -> torch.nn.Module:
        return self.trainer.model


class ResNetModel(BaseModel):
    def __init__(self, params: dict, use_se: bool = False):
        p = copy.deepcopy(params)
        p["architecture"] = "resnet_se" if use_se else "resnet"
        super().__init__(p)
        self.use_se = use_se
        self.trainer = ResNetTrainer(self.params, use_se=use_se)

    def load_model(self):
        self.trainer.load_model(load_optimizer=False)
        self.trainer.model.eval()

    def predict(self, image_path: str) -> PredictionResult:
        self.trainer.model.eval()
        size = int(self.params.get("size", 224))
        
        # Đọc và resize ảnh đúng chuẩn từ thư viện DataLoader (đã loại bỏ F.interpolate tại runtime)
        x = torch.tensor(new_image(image_path, size), dtype=torch.float32).view(-1, 3, size, size)
        x = x.to(self.trainer.device)
        
        with torch.no_grad():
            logits = self.trainer.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            
        cat_prob = float(probs[0].item())
        dog_prob = float(probs[1].item())
        label = "cat" if cat_prob >= dog_prob else "dog"
        confidence = max(cat_prob, dog_prob)
        return PredictionResult(label=label, confidence=confidence, cat_prob=cat_prob, dog_prob=dog_prob)

    def evaluate(self, val_loader):
        return self.trainer.evaluate(val_loader)

    def torch_model(self) -> torch.nn.Module:
        return self.trainer.model


class ModelManager:
    def __init__(self, base_params: dict):
        self.base_params = base_params
        self._cache: Dict[str, BaseModel] = {}

    def get_model(self, model_key: str) -> BaseModel:
        if model_key in self._cache:
            return self._cache[model_key]

        if model_key == "cnn":
            wrapped = CNNModel(self.base_params, use_se=False)
        elif model_key == "se":
            wrapped = CNNModel(self.base_params, use_se=True)
        elif model_key == "resnet":
            wrapped = ResNetModel(self.base_params, use_se=False)
        elif model_key == "resnet_se":
            wrapped = ResNetModel(self.base_params, use_se=True)
        else:
            raise ValueError(f"Model key không hợp lệ: {model_key}")

        try:
            _ensure_checkpoint_available(model_key, self.base_params)
            wrapped.load_model()
        except FileNotFoundError as e:
            raise ValueError(
                f"Chưa tìm thấy checkpoint cho mô hình {model_key}. Vui lòng huấn luyện mô hình này trước hoặc kiểm tra Google Drive." 
            ) from e
        except RuntimeError as e:
            raise ValueError(str(e)) from e
        except Exception as e:
            raise ValueError(f"Lỗi khi tải mô hình {model_key}: {e}") from e

        self._cache[model_key] = wrapped
        return wrapped


class Predictor:
    def __init__(self, manager: ModelManager):
        self.manager = manager

    def predict(self, model_key: str, image_path: str) -> PredictionResult:
        wrapped = self.manager.get_model(model_key)
        return wrapped.predict(image_path)


class HeatmapGenerator:
    def generate(
        self,
        wrapped_model: BaseModel,
        image_path: str,
        input_size: int,
        device: str,
        original_weight: float = 0.6,
        heatmap_weight: float = 0.4,
    ):
        return gradcam_visualization(
            wrapped_model.torch_model(),
            image_path,
            input_size=input_size,
            device=device,
            original_weight=original_weight,
            heatmap_weight=heatmap_weight,
        )
