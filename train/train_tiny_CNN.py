import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_NAMES = ("background", "projectile")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "train" / "config" / "train.yaml"


def resolve_project_path(path_like):
    path = Path(str(path_like))
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)

def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg):
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_cfg == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_cfg)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================
# 1. Tiny CNN模型（低占用）
# ======================
class TinyCNN(nn.Module):
    def __init__(self, in_channels=1, channels=(8, 16, 32)):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.conv3 = nn.Conv2d(c2, c3, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(c3, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32 -> 16
        x = self.pool(F.relu(self.conv2(x)))  # 16 -> 8
        x = F.relu(self.conv3(x))             # 8 x 8
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        return self.fc(x)  # logits


class Augmentor:
    def __init__(self, aug_cfg):
        self.enable = aug_cfg.get("enable", True)

        self.motion_cfg = aug_cfg.get("motion_blur", {})
        self.noise_cfg = aug_cfg.get("gaussian_noise", {})
        self.brightness_cfg = aug_cfg.get("brightness", {})
        self.flip_cfg = aug_cfg.get("random_flip", {})
        self.rotate_cfg = aug_cfg.get("random_rotate", {})

    @staticmethod
    def add_motion_blur(img, kernel_sizes):
        k = int(random.choice(kernel_sizes))
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0
        kernel /= k
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def add_noise(img, mean, std):
        noise = np.random.normal(mean, std, img.shape)
        img = img.astype(np.float32) + noise.astype(np.float32)
        return np.clip(img, 0, 255).astype(np.uint8)

    @staticmethod
    def random_brightness(img, vmin, vmax):
        alpha = random.uniform(vmin, vmax)
        img = img.astype(np.float32) * alpha
        return np.clip(img, 0, 255).astype(np.uint8)

    def __call__(self, img):
        if not self.enable:
            return img

        if self.motion_cfg.get("enable", False):
            if random.random() < float(self.motion_cfg.get("prob", 0.5)):
                kernel_sizes = self.motion_cfg.get("kernel_sizes", [3, 5, 7])
                img = self.add_motion_blur(img, kernel_sizes)

        if self.noise_cfg.get("enable", False):
            if random.random() < float(self.noise_cfg.get("prob", 0.5)):
                mean = float(self.noise_cfg.get("mean", 0.0))
                std = float(self.noise_cfg.get("std", 10.0))
                img = self.add_noise(img, mean, std)

        if self.brightness_cfg.get("enable", False):
            if random.random() < float(self.brightness_cfg.get("prob", 0.5)):
                vmin = float(self.brightness_cfg.get("min", 0.7))
                vmax = float(self.brightness_cfg.get("max", 1.3))
                img = self.random_brightness(img, vmin, vmax)

        if self.flip_cfg.get("enable", False):
            if random.random() < float(self.flip_cfg.get("prob", 0.5)):
                use_h = bool(self.flip_cfg.get("horizontal", True))
                use_v = bool(self.flip_cfg.get("vertical", False))
                if use_h and use_v:
                    flip_code = -1
                elif use_h:
                    flip_code = 1
                elif use_v:
                    flip_code = 0
                else:
                    flip_code = None
                if flip_code is not None:
                    img = cv2.flip(img, flip_code)

        if self.rotate_cfg.get("enable", False):
            if random.random() < float(self.rotate_cfg.get("prob", 0.5)):
                min_angle = float(self.rotate_cfg.get("min", -3.0))
                max_angle = float(self.rotate_cfg.get("max", 3.0))
                angle = random.uniform(min_angle, max_angle)
                h, w = img.shape[:2]
                matrix = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, 1.0)
                img = cv2.warpAffine(
                    img,
                    matrix,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )

        return img


# ======================
# 2. 数据集
# ======================
class ProjectileDataset(Dataset):
    def __init__(self, root, input_size, train, augmentor, normalization_cfg):
        self.root = root
        self.input_size = int(input_size)
        self.train = train
        self.augmentor = augmentor

        self.norm_enable = normalization_cfg.get("enable", False)
        self.norm_mean = float(normalization_cfg.get("mean", 0.5))
        self.norm_std = float(normalization_cfg.get("std", 0.5))
        if self.norm_std <= 0:
            raise ValueError("normalization.std must be > 0.")

        self.samples = []
        for label, cls in enumerate(CLASS_NAMES):
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                raise FileNotFoundError(f"Class folder not found: {cls_path}")
            for file in sorted(os.listdir(cls_path)):
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    self.samples.append((os.path.join(cls_path, file), label))

        if not self.samples:
            raise RuntimeError(f"No images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img = cv2.resize(img, (self.input_size, self.input_size))
        if self.train:
            img = self.augmentor(img)

        img = img.astype(np.float32) / 255.0
        if self.norm_enable:
            img = (img - self.norm_mean) / self.norm_std
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), torch.tensor([float(label)], dtype=torch.float32)


def build_optimizer(model, cfg):
    opt_cfg = cfg.get("optimizer", {})
    opt_type = str(opt_cfg.get("type", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    if opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer, cfg, epochs):
    sch_cfg = cfg.get("scheduler", {})
    if not sch_cfg.get("enable", False):
        return None

    sch_type = str(sch_cfg.get("type", "step")).lower()
    if sch_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    step_size = int(sch_cfg.get("step_size", 10))
    gamma = float(sch_cfg.get("gamma", 0.5))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)


def get_effective_pos_weight(dataset, loss_cfg):
    loss_type = str(loss_cfg.get("type", "bce_logits")).lower()
    if loss_type != "bce_logits":
        raise ValueError("Only `loss.type: bce_logits` is supported for binary classification.")

    if bool(loss_cfg.get("auto_pos_weight", False)):
        num_pos = sum(label for _, label in dataset.samples)
        num_neg = len(dataset.samples) - num_pos
        if num_pos == 0:
            print("Warning: no positive samples in train set, using pos_weight=1.0")
            return 1.0

        raw = float(num_neg) / float(num_pos)
        min_w = float(loss_cfg.get("min_pos_weight", 1.0))
        max_w = float(loss_cfg.get("max_pos_weight", 10.0))
        pos_weight = float(np.clip(raw, min_w, max_w))
        print(f"Auto pos_weight: raw={raw:.4f}, clipped={pos_weight:.4f}")
        return pos_weight

    return float(loss_cfg.get("pos_weight", 1.0))


def compute_metrics_from_probs(probs, labels, threshold):
    preds = (probs >= threshold).astype(np.float32)
    labels = labels.astype(np.float32)

    tp = float(((preds == 1) & (labels == 1)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = float((preds == labels).mean())
    return acc, f1


def pick_best_threshold(probs, labels, threshold_cfg):
    use_dynamic = threshold_cfg.get("use_dynamic", False)
    default_t = float(threshold_cfg.get("default", 0.5))

    if not use_dynamic:
        acc, f1 = compute_metrics_from_probs(probs, labels, default_t)
        return default_t, acc, f1

    low, high = threshold_cfg.get("recommended_range", [0.6, 0.8])
    low = float(low)
    high = float(high)
    if low > high:
        low, high = high, low

    best_t = default_t
    best_acc = -1.0
    best_f1 = -1.0
    for threshold in np.linspace(low, high, num=31):
        acc, f1 = compute_metrics_from_probs(probs, labels, float(threshold))
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and acc > best_acc):
            best_t = float(threshold)
            best_acc = acc
            best_f1 = f1
    return best_t, best_acc, best_f1


def merge_cfg_with_checkpoint(cfg_dict, ckpt_dict, key):
    merged = dict(cfg_dict)
    if isinstance(ckpt_dict, dict):
        ckpt_value = ckpt_dict.get(key, {})
        if isinstance(ckpt_value, dict):
            merged.update(ckpt_value)
    return merged


def resolve_infer_threshold(cfg, ckpt_threshold):
    threshold_cfg = cfg.get("threshold", {})
    if threshold_cfg.get("use_dynamic", False):
        return float(ckpt_threshold)
    return float(threshold_cfg.get("default", ckpt_threshold))


def evaluate(model, data_loader, criterion, device, threshold_cfg):
    model.eval()
    total_loss = 0.0
    probs_all = []
    labels_all = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)

            probs = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy()
            labels_np = labels.squeeze(1).detach().cpu().numpy()
            probs_all.append(probs)
            labels_all.append(labels_np)

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    threshold, acc, f1 = pick_best_threshold(probs_all, labels_all, threshold_cfg)
    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss, acc, f1, threshold


def save_checkpoint(path, model, threshold, cfg):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    ckpt = {
        "state_dict": model.state_dict(),
        "threshold": float(threshold),
        "model_cfg": cfg.get("model", {}),
        "normalization_cfg": cfg.get("normalization", {}),
        "class_names": CLASS_NAMES,
    }
    torch.save(ckpt, path)


def load_model_from_checkpoint(weights_path, device, cfg):
    ckpt = torch.load(weights_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model_cfg = merge_cfg_with_checkpoint(cfg.get("model", {}), ckpt, "model_cfg")
        in_channels = int(model_cfg.get("in_channels", 1))
        channels = tuple(model_cfg.get("channels", [8, 16, 32]))
        model = TinyCNN(in_channels=in_channels, channels=channels)
        model.load_state_dict(ckpt["state_dict"])
        threshold = float(ckpt.get("threshold", cfg.get("threshold", {}).get("default", 0.5)))
        meta = ckpt
    else:
        # 兼容旧版只保存了 state_dict 的权重文件
        model_cfg = dict(cfg.get("model", {}))
        in_channels = int(model_cfg.get("in_channels", 1))
        channels = tuple(model_cfg.get("channels", [8, 16, 32]))
        model = TinyCNN(in_channels=in_channels, channels=channels)
        model.load_state_dict(ckpt)
        threshold = float(cfg.get("threshold", {}).get("default", 0.5))
        meta = {}

    model = model.to(device)
    model.eval()
    return model, threshold, meta, model_cfg


class TinyCNNInferencer:
    def __init__(self, cfg):
        train_cfg = cfg.get("training", {})
        self.device = get_device(str(train_cfg.get("device", "auto")).lower())
        self.weights_path = resolve_project_path(train_cfg.get("weights_path", "model/pellet_cls.pth"))

        self.model, ckpt_threshold, ckpt_meta, model_cfg = load_model_from_checkpoint(
            self.weights_path,
            self.device,
            cfg,
        )
        self.input_size = int(model_cfg.get("input_size", cfg.get("model", {}).get("input_size", 32)))
        self.norm_cfg = merge_cfg_with_checkpoint(cfg.get("normalization", {}), ckpt_meta, "normalization_cfg")
        self.threshold = resolve_infer_threshold(cfg, ckpt_threshold)

        if float(self.norm_cfg.get("std", 0.5)) <= 0:
            raise ValueError("normalization.std must be > 0 for inference.")

    def _preprocess(self, img):
        if img is None:
            raise ValueError("Input image is None.")

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (self.input_size, self.input_size)).astype(np.float32) / 255.0
        if self.norm_cfg.get("enable", False):
            mean = float(self.norm_cfg.get("mean", 0.5))
            std = float(self.norm_cfg.get("std", 0.5))
            img = (img - mean) / std

        return np.expand_dims(img, axis=0)

    def predict_batch(self, roi_images):
        if isinstance(roi_images, np.ndarray):
            if roi_images.ndim in (2, 3):
                roi_images = [roi_images]
            elif roi_images.ndim == 4:
                roi_images = [roi_images[i] for i in range(roi_images.shape[0])]
            else:
                raise ValueError("roi_images ndarray must be 2D, 3D or 4D.")

        if len(roi_images) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.uint8)

        batch = np.stack([self._preprocess(img) for img in roi_images], axis=0).astype(np.float32)
        tensor = torch.from_numpy(batch).to(self.device)

        with torch.no_grad():
            probs = torch.sigmoid(self.model(tensor)).squeeze(1).detach().cpu().numpy().astype(np.float32)
        preds = (probs >= self.threshold).astype(np.uint8)
        return probs, preds

    def predict_one(self, img):
        probs, preds = self.predict_batch([img])
        return float(probs[0]), int(preds[0])


# ======================
# 3. 训练函数
# ======================
def train(cfg):
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    ds_cfg = cfg.get("dataset", {})
    loss_cfg = cfg.get("loss", {})
    strategy_cfg = cfg.get("training_strategy", {})
    early_cfg = strategy_cfg.get("early_stopping", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    device = get_device(str(train_cfg.get("device", "auto")).lower())
    deterministic = bool(train_cfg.get("deterministic", False))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic

    input_size = int(model_cfg.get("input_size", 32))
    in_channels = int(model_cfg.get("in_channels", 1))
    num_classes = int(model_cfg.get("num_classes", 1))
    channels = tuple(model_cfg.get("channels", [8, 16, 32]))
    if num_classes != 1:
        raise ValueError("This script is for binary classification only, so model.num_classes must be 1.")

    train_path = resolve_project_path(ds_cfg.get("train_path", "train/dataset/train"))
    val_path = resolve_project_path(ds_cfg.get("val_path", "train/dataset/val"))
    shuffle = bool(ds_cfg.get("shuffle", True))

    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 30))
    num_workers = int(train_cfg.get("num_workers", 0))
    weights_path = resolve_project_path(train_cfg.get("weights_path", "model/pellet_cls.pth"))

    augmentor = Augmentor(cfg.get("augmentation", {}))
    normalization_cfg = cfg.get("normalization", {})

    train_dataset = ProjectileDataset(
        root=train_path,
        input_size=input_size,
        train=True,
        augmentor=augmentor,
        normalization_cfg=normalization_cfg,
    )
    val_dataset = ProjectileDataset(
        root=val_path,
        input_size=input_size,
        train=False,
        augmentor=augmentor,
        normalization_cfg=normalization_cfg,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    model = TinyCNN(in_channels=in_channels, channels=channels).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, epochs)

    save_best_only = bool(strategy_cfg.get("save_best_only", True))
    early_stop_enable = bool(early_cfg.get("enable", True))
    early_patience = int(early_cfg.get("patience", 5))
    log_interval = max(1, int(strategy_cfg.get("log_interval", 10)))
    print_batch_loss = bool(cfg.get("debug", {}).get("print_batch_loss", False))

    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    print(f"Model params: {count_params(model)}")
    print(f"Input: {input_size}x{input_size}, channels: {channels}")

    pos_weight = get_effective_pos_weight(train_dataset, loss_cfg)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    print(f"Using pos_weight: {pos_weight:.4f}")

    best_acc = -1.0
    best_f1 = -1.0
    best_threshold = float(cfg.get("threshold", {}).get("default", 0.5))
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, (imgs, labels) in enumerate(train_loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            if print_batch_loss and (step % log_interval == 0 or step == len(train_loader)):
                print(f"Epoch {epoch + 1:02d} Step {step:04d} Loss {loss.item():.4f}")

        train_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1, val_threshold = evaluate(
            model,
            val_loader,
            criterion,
            device,
            cfg.get("threshold", {}),
        )

        if scheduler is not None:
            scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1:02d}/{epochs:02d} | "
            f"LR {lr_now:.6f} | Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
            f"Val F1 {val_f1:.4f} | Thr {val_threshold:.3f}"
        )

        improved = (val_f1 > best_f1) or (np.isclose(val_f1, best_f1) and val_acc > best_acc)
        if improved:
            best_acc = val_acc
            best_f1 = val_f1
            best_threshold = val_threshold
            no_improve_epochs = 0
            save_checkpoint(weights_path, model, best_threshold, cfg)
            print(
                f"Saved best model to: {weights_path} "
                f"(Val F1 {best_f1:.4f}, Val Acc {best_acc:.4f}, Thr {best_threshold:.3f})"
            )
        else:
            no_improve_epochs += 1
            if not save_best_only:
                save_checkpoint(weights_path, model, val_threshold, cfg)
                print(f"Saved latest model to: {weights_path}")

        if early_stop_enable and no_improve_epochs >= early_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    print(
        f"Training done. Best Val Acc {best_acc:.4f}, "
        f"Best Val F1 {best_f1:.4f}, Best Thr {best_threshold:.3f}"
    )


# ======================
# 4. 导出 ONNX（部署用）
# ======================
def export_onnx(cfg):
    train_cfg = cfg.get("training", {})
    export_cfg = cfg.get("export", {})

    device = get_device(str(train_cfg.get("device", "auto")).lower())
    weights_path = resolve_project_path(train_cfg.get("weights_path", "model/pellet_cls.pth"))
    onnx_path = resolve_project_path(export_cfg.get("onnx_path", "model/pellet_cls.onnx"))
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    model, threshold, _, model_cfg = load_model_from_checkpoint(weights_path, device, cfg)
    input_size = int(model_cfg.get("input_size", cfg.get("model", {}).get("input_size", 32)))
    in_channels = int(model_cfg.get("in_channels", cfg.get("model", {}).get("in_channels", 1)))
    model = model.cpu()

    dummy = torch.randn(1, in_channels, input_size, input_size)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=11,
    )

    if bool(export_cfg.get("simplify", False)):
        try:
            import onnx
            from onnxsim import simplify

            onnx_model = onnx.load(onnx_path)
            sim_model, ok = simplify(onnx_model)
            if ok:
                onnx.save(sim_model, onnx_path)
                print(f"ONNX simplify done: {onnx_path}")
            else:
                print("Warning: onnxsim returned unsuccessful check, keep original ONNX.")
        except Exception as exc:
            print(f"Warning: ONNX simplify skipped ({exc}).")

    print(f"Export ONNX done: {onnx_path} (recommended threshold {threshold:.3f})")


# ======================
# 5. 测试单张图片
# ======================
def infer(cfg, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    inferencer = TinyCNNInferencer(cfg)
    prob, pred = inferencer.predict_one(img)
    print(f"Probability: {prob:.4f} | Threshold: {inferencer.threshold:.3f} | Pred: {pred}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train / export / infer tiny CNN.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config yaml.")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["train", "export", "infer", "all"],
        help="Run mode.",
    )
    parser.add_argument("--image", default="", help="Image path for infer mode.")
    return parser.parse_args()


# ======================
# 6. 主入口
# ======================
if __name__ == "__main__":
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    cfg = load_config(str(config_path))

    if args.mode in ("train", "all"):
        train(cfg)

    if args.mode in ("export", "all"):
        if cfg.get("export", {}).get("onnx", True):
            export_onnx(cfg)

    if args.mode == "infer":
        if not args.image:
            raise ValueError("Please provide --image for infer mode.")
        infer(cfg, args.image)
