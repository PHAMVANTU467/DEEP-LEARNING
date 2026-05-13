"""
heatmap_utils.py
Tiện ích tạo và hiển thị heatmap Grad-CAM cho các model CNN-based (CNN, CNN+SE, ResNet18, ResNet18+SE).
- Không thay đổi kiến trúc model
- Tận dụng CNN/gradcam.py
- Tương thích PyTorch
- Có thể import và dùng độc lập
"""
import torch # Import PyTorch
import numpy as np # Import thư viện Numpy tính toán ma trận
from PIL import Image # Import thư viện xử lý ảnh Pillow
import sys # Import thư viện tương tác hệ thống
import pathlib # Import thư viện xử lý đường dẫn
import torch.nn.functional as F # Import module chứa hàm interpolate
import cv2 # Import thư viện OpenCV
import importlib # Import module hỗ trợ nạp module linh hoạt bằng chuỗi văn bản

import src.core.gradcam as gradcam_mod # Nạp file gradcam.py
GradCAM = gradcam_mod.GradCAM # Lấy class GradCAM
pil_image_to_tensor_rgb = gradcam_mod.pil_image_to_tensor_rgb # Lấy hàm chuyển ảnh sang tensor RGB


# Hàm bóc tách lớp bao ngoài để lấy ra mô hình gốc (torch.nn.Module)
def _unwrap_torch_model(model_obj):
    """Return torch.nn.Module from common wrappers used in this project."""
    if isinstance(model_obj, torch.nn.Module):
        return model_obj # Nếu là model chuẩn PyTorch thì trả về ngay
    # Nếu là Wrapper Controller, bóc lấy thuộc tính `.model` bên trong
    if hasattr(model_obj, 'model') and isinstance(model_obj.model, torch.nn.Module):
        return model_obj.model
    raise TypeError("Model object không phải torch.nn.Module hoặc wrapper có thuộc tính '.model'.")


# Hàm chuẩn bị ảnh đầu vào (tiền xử lý) cho mô hình
def _prepare_input_tensor(image_path, input_size, device, model_obj):
    """Prepare input tensor with preprocessing aligned to training pipeline."""
    # Đọc ảnh RGB, đưa về tensor chuẩn hóa (theo quy chuẩn hàm pil_image_to_tensor_rgb)
    tensor = pil_image_to_tensor_rgb(image_path, input_size).to(device)
    # Kiểm tra xem mô hình gốc có chứa thuộc tính 'layer4' không (đặc trưng của ResNet)
    core_model = _unwrap_torch_model(model_obj)
    is_resnet = hasattr(core_model, 'layer4') or (hasattr(core_model, 'resnet') and hasattr(core_model.resnet, 'layer4'))
    if is_resnet:
        # Nếu là ResNet, nội suy (phóng to) tensor ảnh lên kích thước 224x224
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    return tensor


# Hàm tự động chọn lớp Convolution cuối cùng để trích xuất Grad-CAM
def get_target_layer(model):
    """Select Grad-CAM target layer with architecture-aware defaults."""
    # Với kiến trúc ResNet (torchvision): Lấy lớp Convolution cuối cùng của block cuối cùng
    if hasattr(model, 'layer4') and len(model.layer4) > 0:
        last_block = model.layer4[-1]
        if hasattr(last_block, 'conv2'):
            return last_block.conv2
        return last_block
    elif hasattr(model, 'resnet') and hasattr(model.resnet, 'layer4') and len(model.resnet.layer4) > 0:
        last_block = model.resnet.layer4[-1]
        if hasattr(last_block, 'conv2'):
            return last_block.conv2
        return last_block
        
    # Với kiến trúc CNN cơ bản tự code: Lấy lớp conv4 (nếu có)
    if hasattr(model, 'conv4'):
        return model.conv4
    if hasattr(model, 'conv3'):
        return model.conv3

    # Fallback: Quét duyệt tìm lớp Conv2d nằm cuối cùng trong mạng
    target = None
    for _, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target = module
    if target is None:
        raise ValueError("Không tìm thấy lớp phù hợp để Grad-CAM.")
    return target

# Hàm chính để tạo dữ liệu ma trận nhiệt (heatmap)
def generate_gradcam_heatmap(model, image_path, input_size=50, device='cpu', target_class=None):
    """
    Sinh heatmap Grad-CAM cho 1 ảnh và 1 model.
    Trả về: heatmap (np.ndarray, [0,1]), tensor input, predicted class, confidence
    """
    core_model = _unwrap_torch_model(model) # Bóc model
    core_model.eval() # Chuyển sang Eval

    # Load ảnh và chuyển về tensor (đúng resize/normalize với training hiện tại)
    input_tensor = _prepare_input_tensor(image_path, input_size, device, model)
    # Tìm layer mục tiêu đúng kiến trúc
    target_layer = get_target_layer(core_model)
    # Khởi tạo object GradCAM
    gradcam = GradCAM(core_model, target_layer=target_layer)
    # Dự đoán (không dùng no_grad để giữ tương thích đầy đủ cho thuật toán đạo hàm của Grad-CAM)
    output = core_model(input_tensor)
    if isinstance(output, tuple):
        output = output[0] # Lấy tensor điểm (nếu trả về tuple)
    probs = torch.softmax(output, dim=1) # Chuyển sang xác suất
    pred_class = int(torch.argmax(probs, dim=1).item()) # Chọn nhãn tin cậy nhất
    confidence = float(probs[0, pred_class].item()) # Mức độ tin cậy
    # Sinh heatmap (Hàm generate tự bật requires_grad=True và gọi backward)
    heatmap = gradcam.generate(input_tensor, target_class=pred_class)
    # Chuẩn hóa min-max an toàn để giá trị luôn nằm trong khoảng [0, 1]
    h_min = float(np.min(heatmap))
    h_max = float(np.max(heatmap))
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8) # +1e-8 chống chia cho 0
    return heatmap, input_tensor, pred_class, confidence

# Hàm hoà trộn (overlay) bản đồ nhiệt lên trên ảnh gốc để hiển thị
def blend_heatmap(image_path, heatmap, original_weight=0.6, heatmap_weight=0.4):
    """Overlay heatmap bằng OpenCV (applyColorMap + addWeighted)."""
    # Đọc ảnh gốc bằng OpenCV hệ màu BGR
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    # Kích thước ảnh gốc
    h, w = original_bgr.shape[:2]
    # Phóng to heatmap [0, 1] bằng nội suy bậc 3 cho mượt
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    # Đưa về thang điểm [0, 255] hệ uint8
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap_resized, 0.0, 1.0))
    # Phủ màu JET (xanh biển -> xanh lá -> vàng -> đỏ) cho bản đồ nhiệt
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    # Hoà trộn ảnh gốc và ảnh nhiệt theo tỉ lệ truyền vào (ví dụ 60% gốc - 40% nhiệt)
    overlay_bgr = cv2.addWeighted(original_bgr, float(original_weight), heatmap_color, float(heatmap_weight), 0)

    # Convert BGR -> RGB để thư viện PIL và Streamlit có thể hiển thị màu chuẩn
    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb) # Trả về ảnh kiểu PIL

# Hàm tiện ích bọc lại 2 hàm phía trên, gọi một lần được cả ảnh kết quả
def gradcam_visualization(
    model,
    image_path,
    input_size=50,
    device='cpu',
    original_weight=0.6,
    heatmap_weight=0.4,
):
    """
    Sinh heatmap và overlay cho 1 ảnh và model.
    Trả về:
        - Ảnh gốc (PIL)
        - Ảnh overlay heatmap (PIL)
        - predicted class, confidence
    """
    # Tính toán ra mảng 2D biểu diễn Heatmap
    heatmap, _, pred_class, confidence = generate_gradcam_heatmap(model, image_path, input_size, device)
    # Lấy mảng 2D đè lên ảnh gốc để ra ảnh có màu đỏ xanh (overlay)
    overlay_img = blend_heatmap(
        image_path,
        heatmap,
        original_weight=original_weight,
        heatmap_weight=heatmap_weight,
    )
    # Tải cả ảnh gốc (không bị đè nhiệt) để UI hiển thị so sánh
    orig_img = Image.open(image_path).convert('RGB')
    return orig_img, overlay_img, pred_class, confidence
