import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        
        self.target_layer = target_layer
        if self.target_layer is None:
            self.target_layer = self._find_target_layer()

        if self.target_layer is None:
            raise ValueError('Không tìm thấy layer phù hợp cho GradCAM.')

        self.activations = None
        self.gradients = None
        self.handlers = []

        # Hook registration
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.handlers.append(self.target_layer.register_forward_hook(forward_hook))
        try:
            self.handlers.append(self.target_layer.register_full_backward_hook(backward_hook))
        except AttributeError:
            self.handlers.append(self.target_layer.register_backward_hook(backward_hook))

    def _find_target_layer(self):
        """Tự động dò tìm layer phù hợp nhất cho GradCAM"""
        # ResNet18
        if hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        
        # ResNet18 + SE (Wrapper)
        if hasattr(self.model, 'resnet') and hasattr(self.model.resnet, 'layer4'):
            return self.model.resnet.layer4[-1]
        
        # CNN / SE_CNN
        # Tìm Conv2d cuối cùng trước Classifier
        target = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                target = module
        return target

    def remove_hooks(self):
        """Xóa hooks để tránh rò rỉ bộ nhớ (Memory Leak)."""
        for handle in self.handlers:
            handle.remove()
        self.handlers = []

    def generate(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad = True

        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        if isinstance(outputs, tuple):
            scores = outputs[0]
        else:
            scores = outputs

        if target_class is None:
            target_class = int(torch.argmax(scores, dim=1).item())

        score = scores[0, target_class]
        
        # retain_graph=False giúp dọn dẹp RAM ngay sau khi tính đạo hàm xong
        score.backward(retain_graph=False)

        grads = self.gradients
        activations = self.activations

        if grads is None or activations is None:
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))

        # Global Average Pooling gradients để tính trọng số alpha
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        
        # Kết hợp trọng số với activations (Linear combination)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU để chỉ giữ lại các feature có ảnh hưởng TÍCH CỰC đến class đó
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()
        
        # Dọn dẹp RAM cục bộ
        self.gradients = None
        self.activations = None
        self.model.zero_grad()
        if input_tensor.grad is not None:
            input_tensor.grad.detach_()
            input_tensor.grad.zero_()

        # Chuẩn hóa về [0, 1]
        cam_min = cam_np.min()
        cam_max = cam_np.max()
        if cam_max - cam_min > 1e-7:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
            
        return cam_np

    def __del__(self):
        self.remove_hooks()


def overlay_heatmap_on_image(image_path, heatmap, output_path=None, alpha=0.5, colormap='jet'):
    img = Image.open(image_path).convert('RGB')
    img_w, img_h = img.size

    # Resize heatmap vừa khít với ảnh gốc
    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize((img_w, img_h), resample=Image.BILINEAR)

    try:
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        colored = cmap(np.array(heatmap_img) / 255.0)
        colored = np.uint8(colored[:, :, :3] * 255)
        heatmap_color = Image.fromarray(colored)
    except Exception:
        arr = np.array(heatmap_img)
        colored = np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=2)
        heatmap_color = Image.fromarray(np.uint8(colored))

    overlay = Image.blend(img, heatmap_color, alpha=alpha)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        overlay.save(output_path)

    return overlay

def pil_image_to_tensor_rgb(image_path, size):
    img = Image.open(image_path).convert('RGB')
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(img).unsqueeze(0)
    return tensor
