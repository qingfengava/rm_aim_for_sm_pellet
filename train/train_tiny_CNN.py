import argparse
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
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


def _deep_merge_dict(base, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def _load_detector_pipeline_cfg(cfg):
    defaults = {
        "motion": {
            "gaussian_ksize": 3,
            "gaussian_sigma": 0.8,
            "diff_threshold": 18,
            "diff_threshold_min": 10,
            "diff_threshold_max": 35,
            "morph_kernel": 3,
            "morph_iters": 1,
            "area_min": 3,
            "area_max": 120,
            "ratio_max": 4.0,
            "max_candidates": 16,
        },
        "roi": {
            "output_size": int(cfg.get("model", {}).get("input_size", 32)),
            "size_scale": 2.2,
            "min_crop": 20,
            "max_crop": 48,
        },
        "inference": {
            "max_candidates": 8,
        },
    }

    pellet_cfg_path = PROJECT_ROOT / "config" / "pellet.yaml"
    if pellet_cfg_path.exists():
        with open(pellet_cfg_path, "r", encoding="utf-8") as f:
            pellet_cfg = yaml.safe_load(f) or {}
        for key in ("motion", "roi", "inference"):
            if isinstance(pellet_cfg.get(key), dict):
                _deep_merge_dict(defaults[key], pellet_cfg[key])

    user_pipe_cfg = cfg.get("detector_pipeline", {})
    if isinstance(user_pipe_cfg, dict):
        for key in ("motion", "roi", "inference"):
            if isinstance(user_pipe_cfg.get(key), dict):
                _deep_merge_dict(defaults[key], user_pipe_cfg[key])

    return defaults


def _to_gray_and_blur(frame_bgr, gaussian_ksize, gaussian_sigma):
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    if frame_bgr.ndim == 2:
        gray = frame_bgr.copy()
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    ksize = int(gaussian_ksize)
    if ksize < 1:
        ksize = 1
    if (ksize % 2) == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), float(gaussian_sigma))


def _three_frame_diff_apply(gray_frame, state, high_threshold):
    prev1 = state.get("prev1")
    prev2 = state.get("prev2")

    if prev1 is None:
        state["prev1"] = gray_frame.copy()
        return np.zeros_like(gray_frame)
    if prev2 is None:
        state["prev2"] = prev1.copy()
        state["prev1"] = gray_frame.copy()
        return np.zeros_like(gray_frame)

    d1 = cv2.absdiff(prev2, prev1)
    d2 = cv2.absdiff(prev1, gray_frame)
    and_mask = cv2.bitwise_and(d1, d2)
    _, high_mask = cv2.threshold(d1, int(high_threshold), 255, cv2.THRESH_BINARY)
    combined = np.maximum(and_mask, high_mask)

    state["prev2"] = prev1.copy()
    state["prev1"] = gray_frame.copy()
    return combined


def _binarize_motion(motion_response, threshold_low, threshold_high):
    _, weak = cv2.threshold(motion_response, int(threshold_low), 255, cv2.THRESH_BINARY)
    if int(threshold_high) <= int(threshold_low):
        return weak

    _, strong = cv2.threshold(motion_response, int(threshold_high), 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    strong_dilated = cv2.dilate(strong, kernel)
    linked = cv2.bitwise_and(weak, strong_dilated)
    return cv2.bitwise_or(linked, strong)


def _apply_open(binary_mask, kernel_size, iterations):
    ksize = int(kernel_size)
    if ksize < 1:
        ksize = 1
    if (ksize % 2) == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=max(1, int(iterations)))


def _extract_candidates(binary_mask, gray_frame, motion_response):
    candidates = []
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        if w <= 0 or h <= 0:
            continue
        if x < 0 or y < 0 or (x + w) > gray_frame.shape[1] or (y + h) > gray_frame.shape[0]:
            continue

        roi = (slice(y, y + h), slice(x, x + w))
        gray_mean = float(np.mean(gray_frame[roi]) / 255.0)
        motion_mean = float(np.mean(motion_response[roi]) / 255.0)

        w_f = float(w)
        h_f = float(max(1, h))
        aspect_ratio = max(w_f / h_f, h_f / max(1.0, w_f))
        circularity_proxy = min(w_f, h_f) / max(w_f, h_f)

        candidates.append({
            "bbox": (x, y, w, h),
            "center": (float(centroids[label, 0]), float(centroids[label, 1])),
            "area": area,
            "motion_score": motion_mean,
            "brightness": gray_mean,
            "circularity": circularity_proxy,
            "aspect_ratio": aspect_ratio,
        })
    return candidates


def _filter_and_rank_candidates(candidates, motion_cfg, inference_cfg):
    area_min = int(motion_cfg.get("area_min", 3))
    area_max = int(motion_cfg.get("area_max", 120))
    ratio_max = float(motion_cfg.get("ratio_max", 4.0))
    max_candidates = min(
        int(motion_cfg.get("max_candidates", 16)),
        int(inference_cfg.get("max_candidates", 8)),
    )

    filtered = []
    for c in candidates:
        if c["area"] < area_min or c["area"] > area_max:
            continue
        if c["aspect_ratio"] > ratio_max:
            continue
        filtered.append(c)

    filtered.sort(key=lambda item: (-item["area"], -item["brightness"]))
    return filtered[: max(0, max_candidates)]


def _crop_roi_batch(gray_frame, candidates, roi_cfg):
    output_size = max(1, int(roi_cfg.get("output_size", 32)))
    size_scale = float(roi_cfg.get("size_scale", 2.2))
    min_crop = max(1, int(roi_cfg.get("min_crop", 20)))
    max_crop = max(min_crop, int(roi_cfg.get("max_crop", 48)))

    h, w = gray_frame.shape[:2]
    patches = []
    for candidate in candidates:
        side_f = np.sqrt(float(max(1, candidate["area"]))) * size_scale
        side = int(round(side_f))
        side = int(np.clip(side, min_crop, max_crop))
        half = max(1, side // 2)

        cx, cy = candidate["center"]
        x = int(cx) - half
        y = int(cy) - half
        x = max(0, min(x, w))
        y = max(0, min(y, h))
        rw = max(0, min(side, w - x))
        rh = max(0, min(side, h - y))
        if rw <= 0 or rh <= 0:
            continue

        roi = gray_frame[y:y + rh, x:x + rw]
        patch = cv2.resize(roi, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        patches.append(patch)
    return patches


def _run_detector_pipeline(frame_bgr, state, pipe_cfg):
    motion_cfg = pipe_cfg["motion"]
    roi_cfg = pipe_cfg["roi"]
    inference_cfg = pipe_cfg["inference"]

    gray = _to_gray_and_blur(
        frame_bgr,
        motion_cfg.get("gaussian_ksize", 3),
        motion_cfg.get("gaussian_sigma", 0.8),
    )
    if gray is None:
        return []

    motion = _three_frame_diff_apply(gray, state, motion_cfg.get("diff_threshold_max", 35))
    binary = _binarize_motion(
        motion,
        motion_cfg.get("diff_threshold_min", 10),
        motion_cfg.get("diff_threshold", 18),
    )
    mask = _apply_open(
        binary,
        motion_cfg.get("morph_kernel", 3),
        motion_cfg.get("morph_iters", 1),
    )
    candidates = _extract_candidates(mask, gray, motion)
    candidates = _filter_and_rank_candidates(candidates, motion_cfg, inference_cfg)
    return _crop_roi_batch(gray, candidates, roi_cfg)


def _collect_sequences(source_root):
    root = Path(source_root)
    if not root.exists():
        return []

    if root.is_file():
        suffix = root.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return [("video", root)]
        if suffix in IMAGE_EXTENSIONS:
            return [("images", [root])]
        return []

    sequences = []
    for video_path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS):
        sequences.append(("video", video_path))

    image_paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    grouped = {}
    for image_path in image_paths:
        grouped.setdefault(image_path.parent, []).append(image_path)
    for parent in sorted(grouped.keys()):
        sequences.append(("images", grouped[parent]))
    return sequences


def _iter_frames(sequence):
    kind, payload = sequence
    if kind == "video":
        cap = cv2.VideoCapture(str(payload))
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield frame
        finally:
            cap.release()
        return

    for image_path in payload:
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is not None:
            yield frame


def _ensure_dataset_layout(root_dir):
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(root_dir, cls), exist_ok=True)


def _prepare_split_with_pipeline(raw_split_root, out_split_root, ds_cfg, pipe_cfg):
    _ensure_dataset_layout(out_split_root)

    frame_stride = max(1, int(ds_cfg.get("pipeline_frame_stride", 1)))
    max_per_frame_pos = max(1, int(ds_cfg.get("pipeline_pos_max_per_frame", 16)))
    max_per_frame_bg = max(1, int(ds_cfg.get("pipeline_bg_max_per_frame", 16)))
    out_ext = str(ds_cfg.get("pipeline_output_ext", "png")).lower().lstrip(".")

    counts = {cls: 0 for cls in CLASS_NAMES}
    for cls in CLASS_NAMES:
        class_raw_root = Path(raw_split_root) / cls
        if not class_raw_root.exists():
            print(f"Warning: raw class path not found, skip: {class_raw_root}")
            continue

        sequences = _collect_sequences(class_raw_root)
        if not sequences:
            print(f"Warning: no media found under {class_raw_root}")
            continue

        for seq_idx, sequence in enumerate(sequences):
            state = {"prev1": None, "prev2": None}
            for frame_idx, frame in enumerate(_iter_frames(sequence)):
                if (frame_idx % frame_stride) != 0:
                    continue

                patches = _run_detector_pipeline(frame, state, pipe_cfg)
                if not patches:
                    continue

                limit = max_per_frame_pos if cls == "projectile" else max_per_frame_bg
                for patch_idx, patch in enumerate(patches[:limit]):
                    filename = f"{cls}_s{seq_idx:04d}_f{frame_idx:06d}_p{patch_idx:02d}.{out_ext}"
                    out_path = os.path.join(out_split_root, cls, filename)
                    cv2.imwrite(out_path, patch)
                    counts[cls] += 1

    print(
        f"[detector_pipeline] split={Path(out_split_root).name} "
        f"projectile={counts['projectile']} background={counts['background']}"
    )


def _resolve_dataset_paths(cfg, prepare_with_pipeline):
    ds_cfg = cfg.get("dataset", {})
    train_path = resolve_project_path(ds_cfg.get("train_path", "train/dataset/train"))
    val_path = resolve_project_path(ds_cfg.get("val_path", "train/dataset/val"))

    if not bool(ds_cfg.get("use_detector_pipeline", False)):
        return train_path, val_path

    raw_train_root = resolve_project_path(ds_cfg.get("raw_train_path", "train/dataset_raw/train"))
    raw_val_root = resolve_project_path(ds_cfg.get("raw_val_path", "train/dataset_raw/val"))
    processed_root = resolve_project_path(ds_cfg.get("processed_path", "train/dataset"))
    train_path = os.path.join(processed_root, "train")
    val_path = os.path.join(processed_root, "val")

    if prepare_with_pipeline:
        if not os.path.isdir(raw_train_root):
            raise FileNotFoundError(
                f"dataset.use_detector_pipeline=1, but raw_train_path not found: {raw_train_root}"
            )
        if not os.path.isdir(raw_val_root):
            raise FileNotFoundError(
                f"dataset.use_detector_pipeline=1, but raw_val_path not found: {raw_val_root}"
            )

        if bool(ds_cfg.get("overwrite_processed", False)) and os.path.isdir(processed_root):
            shutil.rmtree(processed_root)

        pipe_cfg = _load_detector_pipeline_cfg(cfg)
        _prepare_split_with_pipeline(raw_train_root, train_path, ds_cfg, pipe_cfg)
        _prepare_split_with_pipeline(raw_val_root, val_path, ds_cfg, pipe_cfg)

    return train_path, val_path


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

    train_path, val_path = _resolve_dataset_paths(cfg, prepare_with_pipeline=True)
    shuffle = bool(ds_cfg.get("shuffle", True))

    batch_size = int(train_cfg.get("batch_size", 64))
    epochs = int(train_cfg.get("epochs", 30))
    num_workers = int(train_cfg.get("num_workers", 0))
    weights_path = resolve_project_path(train_cfg.get("weights_path", "model/pellet_cls.pth"))

    aug_cfg = dict(cfg.get("augmentation", {}))
    if bool(ds_cfg.get("use_detector_pipeline", False)) and not bool(ds_cfg.get("keep_augmentation_after_pipeline", False)):
        aug_cfg["enable"] = False
    augmentor = Augmentor(aug_cfg)
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
        choices=["prepare", "train", "export", "infer", "all"],
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

    if args.mode in ("prepare",):
        train_path, val_path = _resolve_dataset_paths(cfg, prepare_with_pipeline=True)
        print(f"Dataset prepared by detector pipeline.\n  train: {train_path}\n  val:   {val_path}")

    if args.mode in ("train", "all"):
        train(cfg)

    if args.mode in ("export", "all"):
        if cfg.get("export", {}).get("onnx", True):
            export_onnx(cfg)

    if args.mode == "infer":
        if not args.image:
            raise ValueError("Please provide --image for infer mode.")
        infer(cfg, args.image)
