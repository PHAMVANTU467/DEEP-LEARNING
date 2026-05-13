from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import copy
import os
from typing import Dict
import importlib

import torch
import torch.nn.functional as F

from src.app.heatmap import gradcam_visualization
import src.models.cnn_trainer as cnn_model_mod
import src.models.resnet_trainer as resnet_mod
import src.core.tools as tools_mod

CNNTrainer = cnn_model_mod.CNNTrainer
ResNetTrainer = resnet_mod.ResNetTrainer
new_image = tools_mod.new_image

@dataclass
class PredictionResult:
    label: str
    confidence: float
    cat_prob: float
    dog_prob: float

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
            wrapped.load_model()
        except FileNotFoundError as e:
            raise ValueError(f"Chưa tìm thấy checkpoint cho mô hình {model_key}. Vui lòng huấn luyện mô hình này trước!") from e
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
