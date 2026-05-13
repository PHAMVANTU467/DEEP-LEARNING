import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import os
import shutil
import warnings
from PIL import Image, ImageFile
from src.core.paths import DATA_DIR, CORRUPTED_DATA_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_train_transform(size: int):
    """
    Data Augmentation for Training (Combats shortcut learning).
    - RandomResizedCrop
    - HorizontalFlip
    - ColorJitter
    - RandomRotation
    - GaussianBlur (simulate focus differences)
    - ToTensor
    - Normalize
    """
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform(size: int):
    """Validation transform: Resize directly or center crop, ToTensor, Normalize"""
    val_size = int(size * 1.14)
    return T.Compose([
        T.Resize(val_size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def safe_loader(path):
    from torchvision.datasets.folder import default_loader
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        img = default_loader(path)
        for warning in w:
            if "Truncated File Read" in str(warning.message):
                os.makedirs(CORRUPTED_DATA_DIR, exist_ok=True)
                parent_dir_name = os.path.basename(os.path.dirname(path))
                filename = f"{parent_dir_name}_{os.path.basename(path)}"
                dest = os.path.join(CORRUPTED_DATA_DIR, filename)
                if not os.path.exists(dest):
                    try: shutil.copy2(path, dest)
                    except Exception: pass
                break
        return img

def load_data_catsVsdogs(params):
    dataset_name = params.get('dataset_name', '')
    train_dir = os.path.join(DATA_DIR, 'processed', dataset_name, 'train')
    val_dir = os.path.join(DATA_DIR, 'processed', dataset_name, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Không tìm thấy thư mục '{train_dir}' và '{val_dir}'. Vui lòng chạy tab Tiền xử lý dữ liệu trước!")

    size = params.get('size', 224)
    train_transform = get_train_transform(size)
    val_transform = get_val_transform(size)

    train_dataset = ImageFolder(root=train_dir, transform=train_transform, loader=safe_loader)
    val_dataset = ImageFolder(root=val_dir, transform=val_transform, loader=safe_loader)

    # Windows multi-processing is unstable in Streamlit, keep num_workers=0 or 2 if safe
    num_workers = 0 

    train_dl = DataLoader(
        train_dataset,
        batch_size=params['batch_size_training'],
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )

    valid_dl = DataLoader(
        val_dataset,
        batch_size=params['batch_size_validation'],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers
    )

    return train_dl, valid_dl

def new_image(path, size):
    """Load image for inference"""
    img = Image.open(path).convert('RGB')
    transform = get_val_transform(size)
    img_tensor = transform(img)
    return img_tensor.numpy()
def size_conv_output(params):
    # Dummy function for backward compatibility with main.py UI variables
    return 256
