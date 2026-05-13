import torch
import torch.nn as nn
from torchvision import models
from src.core.paths import get_checkpoint_dir, get_best_model_path, get_history_path
import os
import json
import gc

class SEBasicBlock(nn.Module):
    """
    Squeeze-and-Excitation cho ResNet BasicBlock.
    Áp dụng chuẩn: f(x) = SE(Conv2(Conv1(x))) + x
    """
    def __init__(self, block):
        super().__init__()
        self.block = block
        try:
            from src.models.se_cnn import SEBlock
            self.se = SEBlock(block.conv2.out_channels, reduction=16)
        except Exception:
            self.se = nn.Identity()

    def forward(self, x):
        identity = x
        if self.block.downsample is not None:
            identity = self.block.downsample(x)

        out = self.block.conv1(x)
        out = self.block.bn1(out)
        out = self.block.relu(out)

        out = self.block.conv2(out)
        out = self.block.bn2(out)
        
        # Inject SE here
        out = self.se(out)

        out += identity
        out = self.block.relu(out)

        return out

def inject_se_into_resnet(model):
    """Thay thế đệ quy tất cả BasicBlock bằng SEBasicBlock"""
    for name, module in model.named_children():
        if isinstance(module, models.resnet.BasicBlock):
            setattr(model, name, SEBasicBlock(module))
        else:
            inject_se_into_resnet(module)

class ResNetTrainer:
    def __init__(self, params, use_se=False):
        self.params = params
        self.use_se = use_se
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet18
        # Dùng weights mặc định thay vì pretrained=True (cũ)
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights)
        
        # Thay thế lớp Fully Connected cuối cùng bằng 2 class (Cats vs Dogs)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 2)
        )
        
        if self.use_se:
            inject_se_into_resnet(self.model)
            
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        lr = self.params.get('lr', 1e-4)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        epochs = self.params.get('epochs', 10)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

    def load_model(self, load_optimizer=True):
        dataset_name = self.params.get('dataset_name', 'dataset_80_20')
        arch = self.params.get('architecture', 'resnet_se' if self.use_se else 'resnet')
        path = get_best_model_path(dataset_name, arch)
        
        if not path.exists():
            raise FileNotFoundError(f"Chưa có checkpoint: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        self.start_epoch = checkpoint.get('epoch', -1) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))

    def save_model(self, epoch, best_loss):
        dataset_name = self.params.get('dataset_name', 'dataset_80_20')
        arch = self.params.get('architecture', 'resnet_se' if self.use_se else 'resnet')
        path = get_best_model_path(dataset_name, arch)
        temp_path = path.with_suffix('.pth.tmp')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': best_loss,
            'params': self.params
        }, temp_path)
        
        temp_path.replace(path)

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True).float()
                y = y.to(self.device, non_blocking=True).long()
                
                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        yhat = self.model(x)
                        loss = self.criterion(yhat, y)
                else:
                    yhat = self.model(x)
                    loss = self.criterion(yhat, y)
                    
                total_loss += loss.item() * x.size(0)
                total_correct += (yhat.argmax(dim=1) == y).sum().item()
                total_samples += x.size(0)

        del x, y, yhat, loss
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return total_loss / total_samples, (total_correct / total_samples) * 100

    def training(self, tr, epoch=None, total_epochs=None, progress_callback=None, stop_requested=None):
        from tqdm import tqdm
        self.model.train()
        total_loss = 0.0
        total_samples = len(tr) * tr.batch_size
        
        pbar = tqdm(total=total_samples, desc=f"Train Epoch {epoch}", leave=False)

        for x, y in tr:
            x = x.to(self.device, non_blocking=True).float()
            y = y.to(self.device, non_blocking=True).long()
            
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    yhat = self.model(x)
                    loss = self.criterion(yhat, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                yhat = self.model(x)
                loss = self.criterion(yhat, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
            total_loss += loss.item()
            
            pbar.update(x.size(0))
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if progress_callback:
                try: progress_callback(epoch, total_epochs, loss.item(), None, None, pbar.n, total_samples)
                except Exception: pass
                
            if stop_requested and stop_requested():
                break

        pbar.close()
        del x, y, yhat, loss
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        return total_loss / len(tr) if len(tr) > 0 else float('inf')

    def converge(self, tr, val, progress_callback=None, stop_requested=None):
        dataset_name = self.params.get('dataset_name', 'dataset_80_20')
        arch = self.params.get('architecture', 'resnet_se' if self.use_se else 'resnet')
        
        history_path = get_history_path(dataset_name, arch)
        history = []
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    history = json.load(f)
            except Exception: pass

        if not hasattr(self, 'start_epoch'):
            self.start_epoch = 0
            self.best_loss = float('inf')
            
        epochs = self.params.get('epochs', 10)
        patience = self.params.get('patience', 2) # Dừng sớm sau 3 epoch không cải thiện
        
        # Phục hồi bộ đếm Early Stopping từ lịch sử cũ nếu có
        self.epochs_no_improve = 0
        if history:
            best_hist_loss = min(h['val_loss'] for h in history)
            best_ep = [h['epoch'] for h in history if h['val_loss'] == best_hist_loss][0]
            last_ep = history[-1]['epoch']
            self.epochs_no_improve = max(0, last_ep - best_ep)
            
        for epoch in range(self.start_epoch, epochs):
            if self.epochs_no_improve >= patience:
                if progress_callback:
                    try: progress_callback(epoch, epochs, 0, 0, 0, 0, 0)
                    except: pass
                break
                
            train_loss = self.training(tr, epoch, epochs, progress_callback, stop_requested)
            
            if stop_requested and stop_requested():
                break
                
            val_loss, val_acc = self.evaluate(val)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr
            })
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=4)
                
            self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_model(epoch, self.best_loss)
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                
            if self.epochs_no_improve >= patience:
                break

def load_resnet_model(params, load_optimizer=False):
    import copy
    p = copy.deepcopy(params)
    p['architecture'] = 'resnet'
    trainer = ResNetTrainer(p, use_se=False)
    try:
        trainer.load_model(load_optimizer=load_optimizer)
    except Exception:
        pass
    return trainer

def load_resnet_se_model(params, load_optimizer=False):
    import copy
    p = copy.deepcopy(params)
    p['architecture'] = 'resnet_se'
    trainer = ResNetTrainer(p, use_se=True)
    try:
        trainer.load_model(load_optimizer=load_optimizer)
    except Exception:
        pass
    return trainer
