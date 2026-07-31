

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from mosaic import PolarizationMosaicPipeline
from model import DualUCNN

try:
    import scipy.io as sio
except ImportError:
    sio = None

try:
    import h5py
except ImportError:
    h5py = None


# ============================= Losses =============================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_tensor(x):
    return torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)


def charb(pred, target, eps=1e-6):
    pred = safe_tensor(pred)
    target = safe_tensor(target)
    return torch.mean(torch.sqrt((pred - target) ** 2 + eps))


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        self.register_buffer("window", self._make_window(window_size))

    @staticmethod
    def _make_window(window_size):
        g = torch.tensor([
            math.exp(-((x - window_size // 2) ** 2) / 2.0)
            for x in range(window_size)
        ], dtype=torch.float32)
        g /= g.sum()
        return (g[:, None] * g[None, :])[None, None]

    def _single_channel(self, x, y):
        x, y = safe_tensor(x), safe_tensor(y)
        w = self.window.to(device=x.device, dtype=x.dtype)
        pad = self.window_size // 2
        c1, c2 = 0.01 ** 2, 0.03 ** 2

        mu_x = F.conv2d(x, w, padding=pad)
        mu_y = F.conv2d(y, w, padding=pad)
        mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
        var_x = F.conv2d(x * x, w, padding=pad) - mu_x2
        var_y = F.conv2d(y * y, w, padding=pad) - mu_y2
        cov_xy = F.conv2d(x * y, w, padding=pad) - mu_xy

        numerator = (2 * mu_xy + c1) * (2 * cov_xy + c2)
        denominator = ((mu_x2 + mu_y2 + c1) * (var_x + var_y + c2)).clamp_min(1e-6)
        return safe_tensor(numerator / denominator).clamp(-1, 1).mean()

    def forward(self, pred, target):
        loss = pred.new_zeros(())
        for channel in range(pred.shape[1]):
            loss += 1.0 - self._single_channel(
                pred[:, channel:channel + 1],
                target[:, channel:channel + 1],
            )
        return loss / pred.shape[1]


def stokes_loss(pred, target):
    pred = safe_tensor(pred).clamp(0, 1)
    target = safe_tensor(target).clamp(0, 1)
    loss = pred.new_zeros(())

    for color in range(3):
        base = color * 4
        i0, i45 = pred[:, base:base + 1], pred[:, base + 1:base + 2]
        i90, i135 = pred[:, base + 2:base + 3], pred[:, base + 3:base + 4]
        g0, g45 = target[:, base:base + 1], target[:, base + 1:base + 2]
        g90, g135 = target[:, base + 2:base + 3], target[:, base + 3:base + 4]

        s0p, s0g = (i0 + i90) / 2, (g0 + g90) / 2
        s1p, s1g = (i0 - i90 + 1) / 2, (g0 - g90 + 1) / 2
        s2p, s2g = (i45 - i135 + 1) / 2, (g45 - g135 + 1) / 2
        dopp = (torch.sqrt((i0 - i90) ** 2 + (i45 - i135) ** 2 + 1e-6)
                / (i0 + i90).clamp_min(1e-6)).clamp(0, 1)
        dopg = (torch.sqrt((g0 - g90) ** 2 + (g45 - g135) ** 2 + 1e-6)
                / (g0 + g90).clamp_min(1e-6)).clamp(0, 1)

        loss += (charb(s0p, s0g) + charb(s1p, s1g)
                 + charb(s2p, s2g) + charb(dopp, dopg)) / 4
    return loss / 3


# ====================== Physical degradation ======================

WATER_PARAMS = {
    "xxx":       {"airlight": [x,x,x], "factor_r": x, "factor_g": x, "factor_b": x, "trans_ratio": x}},
    
BETA_VALUES = [x,x,x]
IDENTITY_TYPE = "t1_identity"
ANGLES = {"0": 0.0, "45": math.pi / 4, "90": math.pi / 2, "135": 3 * math.pi / 4}
POL_DEGREE = 0.3
POL_ANGLE = math.pi / 4


def random_degradation_params():
    water_type = str(np.random.choice(list(WATER_PARAMS) + [IDENTITY_TYPE]))
    if water_type == IDENTITY_TYPE:
        return {"water_type": water_type, "beta": 0.0, "t_is_one": True}
    return {
        "water_type": water_type,
        "beta": float(np.random.choice(BETA_VALUES)),
        "t_is_one": False,
    }


def validation_degradation_params(include_identity=True):
    params = [
        {"water_type": water, "beta": beta, "t_is_one": False}
        for water in WATER_PARAMS
        for beta in BETA_VALUES
    ]
    if include_identity:
        params.append({"water_type": IDENTITY_TYPE, "beta": 0.0, "t_is_one": True})
    return params


def make_depth(height, width, device):
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    depth = torch.sqrt((xx - 0.5) ** 2 + (yy - 0.5) ** 2)
    return (depth / depth.max()).unsqueeze(0).unsqueeze(0)


def transmission_maps(reference, params, depth_map=None):
    batch, _, height, width = reference.shape
    water = WATER_PARAMS[params["water_type"]]
    beta = params["beta"]

    if depth_map is None:
        depth = make_depth(height, width, reference.device).expand(batch, 1, height, width)
    else:
        depth = F.interpolate(
            depth_map.to(reference.device), (height, width),
            mode="bilinear", align_corners=True,
        ).expand(batch, 1, height, width)

    t_r = (torch.exp(-beta * water["factor_r"] * depth) * water["trans_ratio"]).clamp(0.01, 1)
    t_g = (torch.exp(-beta * water["factor_g"] * depth) * water["trans_ratio"]).clamp(0.01, 1)
    t_b = (torch.exp(-beta * water["factor_b"] * depth) * water["trans_ratio"]).clamp(0.01, 1)
    return t_r, t_g, t_b


def angle_airlight(water, angle):
    coeff = 0.5 * (1 + POL_DEGREE * math.cos(2 * (ANGLES[angle] - POL_ANGLE)))
    return [max(0.0, min(1.0, value * coeff)) for value in water["airlight"]]


def degrade_batch(pol_0, pol_45, pol_90, pol_135, params, depth_map=None):
    if params.get("t_is_one", False):
        return pol_0, pol_45, pol_90, pol_135

    water = WATER_PARAMS[params["water_type"]]
    t_r, t_g, t_b = transmission_maps(pol_0, params, depth_map)

    def degrade(image, angle):
        b_r, b_g, b_b = angle_airlight(water, angle)
        return torch.cat([
            image[:, 0:1] * t_r + b_r * (1 - t_r),
            image[:, 1:2] * t_g + b_g * (1 - t_g),
            image[:, 2:3] * t_b + b_b * (1 - t_b),
        ], dim=1).clamp(0, 1)

    return degrade(pol_0, "0"), degrade(pol_45, "45"), degrade(pol_90, "90"), degrade(pol_135, "135")


def physics_restore(underwater_12ch, params, depth_map=None):
    if params.get("t_is_one", False):
        return underwater_12ch

    water = WATER_PARAMS[params["water_type"]]
    t_r, t_g, t_b = transmission_maps(underwater_12ch, params, depth_map)
    output = torch.zeros_like(underwater_12ch)

    for index, angle in enumerate(("0", "45", "90", "135")):
        b_r, b_g, b_b = angle_airlight(water, angle)
        output[:, index:index + 1] = ((underwater_12ch[:, index:index + 1] - b_r * (1 - t_r)) / t_r).clamp(0, 1)
        output[:, 4 + index:5 + index] = ((underwater_12ch[:, 4 + index:5 + index] - b_g * (1 - t_g)) / t_g).clamp(0, 1)
        output[:, 8 + index:9 + index] = ((underwater_12ch[:, 8 + index:9 + index] - b_b * (1 - t_b)) / t_b).clamp(0, 1)
    return output


# ========================= Data processing ========================

def make_12ch(pol_0, pol_45, pol_90, pol_135):
    batch, _, height, width = pol_0.shape
    output = torch.zeros(batch, 12, height, width, dtype=pol_0.dtype, device=pol_0.device)
    for color in range(3):
        base = color * 4
        output[:, base:base + 1] = pol_0[:, color:color + 1]
        output[:, base + 1:base + 2] = pol_45[:, color:color + 1]
        output[:, base + 2:base + 3] = pol_90[:, color:color + 1]
        output[:, base + 3:base + 4] = pol_135[:, color:color + 1]
    return output


def build_polar_input(mosaic_12ch):
    _, _, height, width = mosaic_12ch.shape
    pooled = F.max_pool2d(mosaic_12ch, kernel_size=2, stride=1, padding=1)
    pooled = pooled[:, :, :height, :width]
    down = F.interpolate(pooled, scale_factor=0.5, mode="bilinear", align_corners=False)
    return F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False).clamp(0, 1)


def load_depth_mat(path):
    depth = None
    if sio is not None:
        try:
            depth = np.asarray(sio.loadmat(path)["depth_normalized"], dtype=np.float32)
        except (KeyError, OSError, ValueError):
            pass
    if depth is None and h5py is not None:
        try:
            with h5py.File(path, "r") as file:
                depth = np.asarray(file["depth_normalized"], dtype=np.float32)
                if depth.ndim == 2:
                    depth = depth.T
        except (KeyError, OSError, ValueError):
            pass
    if depth is None:
        return None
    return (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)


class PolarDataset(Dataset):
    FILE_RULES = [
        ("pol_0.png", "pol_45.png", "pol_90.png", "pol_135.png"),
        ("pol_0.jpg", "pol_45.jpg", "pol_90.jpg", "pol_135.jpg"),
        ("0.bmp", "45.bmp", "90.bmp", "135.bmp"),
        ("0.png", "45.png", "90.png", "135.png"),
    ]

    def __init__(self, image_dir, sample_names, transform=None, depth_dir=None):
        self.transform = transform
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.samples = []

        image_dir = Path(image_dir)
        for name in sample_names:
            sample_dir = image_dir / name
            for filenames in self.FILE_RULES:
                paths = [sample_dir / filename for filename in filenames]
                if all(path.exists() for path in paths):
                    self.samples.append((name, paths))
                    break

        if not self.samples:
            raise ValueError(f"No valid samples found in {image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, paths = self.samples[index]
        images = [Image.open(path).convert("RGB") for path in paths]
        if self.transform:
            images = [self.transform(image) for image in images]

        depth = None
        if self.depth_dir:
            depth_path = self.depth_dir / f"{name}_depth.mat"
            if depth_path.exists():
                depth_array = load_depth_mat(str(depth_path))
                if depth_array is not None:
                    depth = torch.from_numpy(depth_array)[None, None]

        return images[0], images[1], images[2], images[3], name, depth


def collate_fn(batch):
    pol_0 = torch.stack([item[0] for item in batch])
    pol_45 = torch.stack([item[1] for item in batch])
    pol_90 = torch.stack([item[2] for item in batch])
    pol_135 = torch.stack([item[3] for item in batch])
    names = [item[4] for item in batch]
    depths = [item[5] for item in batch]

    if all(depth is None for depth in depths):
        depth_batch = None
    else:
        reference = next(depth for depth in depths if depth is not None)
        _, _, height, width = reference.shape
        depths = [depth if depth is not None else torch.zeros(1, 1, height, width) for depth in depths]
        depth_batch = torch.cat(depths, dim=0)
    return pol_0, pol_45, pol_90, pol_135, names, depth_batch


def split_dataset(image_dir, n_train=90, n_val=15):
    all_dirs = sorted(path.name for path in Path(image_dir).iterdir() if path.is_dir())
    if len(all_dirs) < n_train + n_val:
        raise ValueError(
            f"Found {len(all_dirs)} scene folders, but {n_train + n_val} are required."
        )
    train_names = all_dirs[:n_train]
    val_names = all_dirs[n_train:n_train + n_val]
    return train_names, val_names


# ============================= Trainer ============================

class Trainer:
    def __init__(self, config):
        self.config = config
        set_seed(config.get("seed", 42))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = PolarizationMosaicPipeline()
        self.model = DualUCNN(
            rgb_base_ch=config.get("rgb_base_ch", 32),
            polar_base_ch=config.get("polar_base_ch", 32),
            fusion_mid_ch=config.get("fusion_mid_ch", 32),
        ).to(self.device)

        lr = config.get("lr", 2e-4)
        self.optimizer = optim.AdamW([
            {"params": self.model.rgb_params(), "lr": lr},
            {"params": self.model.polar_params(), "lr": config.get("polar_lr", lr)},
            {"params": self.model.fusion_params(), "lr": config.get("fusion_lr", lr)},
        ], lr=lr, weight_decay=config.get("weight_decay", 0.0))

        self.warmup_epochs = config.get("warmup_epochs", 3)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config["num_epochs"] - self.warmup_epochs),
            eta_min=lr * 0.01,
        )

        self.w_rgb = config.get("w_rgb", 0.4)
        self.w_polar = config.get("w_polar_img", 0.4)
        self.w_fused = config.get("w_fused", 1.0)
        self.w_stokes = config.get("w_stokes", 0.2)
        self.w_ssim = config.get("w_ssim", 0.2)
        self.w_air = config.get("w_air", 0.3)
        self.ssim = SSIMLoss().to(self.device)

        self.output_dir = Path(config.get("output_dir", "./outputs"))
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.best_val_psnr = -float("inf")
        self.val_params = validation_degradation_params(config.get("val_include_identity", True))
        self.setup_data()

    def setup_data(self):
        transform = transforms.Compose([
            transforms.Resize((self.config["image_size"], self.config["image_size"])),
            transforms.ToTensor(),
        ])
        train_names, val_names = split_dataset(
            self.config["data_dir"],
            self.config.get("n_train", 90),
            self.config.get("n_val", 15),
        )

        with open(self.output_dir / "train_sample_names.json", "w", encoding="utf-8") as file:
            json.dump(train_names, file, ensure_ascii=False, indent=2)
        with open(self.output_dir / "val_sample_names.json", "w", encoding="utf-8") as file:
            json.dump(val_names, file, ensure_ascii=False, indent=2)

        train_set = PolarDataset(
            self.config["data_dir"], train_names, transform,
            self.config.get("depth_mat_dir"),
        )
        val_set = PolarDataset(
            self.config["data_dir"], val_names, transform,
            self.config.get("depth_mat_dir"),
        )
        loader_args = {
            "num_workers": self.config.get("num_workers", 0),
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available(),
        }
        self.train_loader = DataLoader(
            train_set, batch_size=self.config["batch_size"],
            shuffle=True, drop_last=True, **loader_args,
        )
        self.val_loader = DataLoader(
            val_set, batch_size=self.config.get("val_batch_size", 1),
            shuffle=False, drop_last=False, **loader_args,
        )
        print(f"Device: {self.device}; train: {len(train_set)}; validation: {len(val_set)}")

    def move_batch(self, batch):
        pol_0, pol_45, pol_90, pol_135, names, depth = batch
        pol_0 = pol_0.to(self.device, non_blocking=True)
        pol_45 = pol_45.to(self.device, non_blocking=True)
        pol_90 = pol_90.to(self.device, non_blocking=True)
        pol_135 = pol_135.to(self.device, non_blocking=True)
        if depth is not None:
            depth = depth.to(self.device, non_blocking=True)
        return pol_0, pol_45, pol_90, pol_135, names, depth

    def forward_model(self, pol_0, pol_45, pol_90, pol_135, depth, params):
        d0, d45, d90, d135 = degrade_batch(
            pol_0, pol_45, pol_90, pol_135, params, depth,
        )
        clean_target = make_12ch(pol_0, pol_45, pol_90, pol_135).clamp(0, 1)
        degraded_target = make_12ch(d0, d45, d90, d135).clamp(0, 1)
        _, mosaic_12ch, _ = self.pipeline(d0, d45, d90, d135)
        polar_input = build_polar_input(mosaic_12ch)
        rgb_out, polar_out, fused_out = self.model(mosaic_12ch, polar_input)
        air_out = physics_restore(fused_out.clamp(0, 1), params, depth)
        return rgb_out, polar_out, fused_out, air_out, degraded_target, clean_target

    def training_loss(self, outputs):
        rgb_out, polar_out, fused_out, air_out, degraded_target, clean_target = outputs
        zero = fused_out.new_zeros(())
        loss_rgb = charb(rgb_out, degraded_target) if self.w_rgb > 0 else zero
        loss_polar = charb(polar_out, degraded_target) if self.w_polar > 0 else zero
        loss_fused = charb(fused_out, degraded_target) if self.w_fused > 0 else zero
        loss_stokes = stokes_loss(polar_out, degraded_target) if self.w_stokes > 0 else zero
        loss_ssim = self.ssim(fused_out, degraded_target) if self.w_ssim > 0 else zero
        loss_air = charb(air_out, clean_target) if self.w_air > 0 else zero
        return (
            self.w_rgb * loss_rgb
            + self.w_polar * loss_polar
            + self.w_fused * loss_fused
            + self.w_stokes * loss_stokes
            + self.w_ssim * loss_ssim
            + self.w_air * loss_air
        )

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = []
        progress = tqdm(self.train_loader, desc=f"Train {epoch + 1}/{self.config['num_epochs']}")

        for batch in progress:
            pol_0, pol_45, pol_90, pol_135, _, depth = self.move_batch(batch)
            params = random_degradation_params()
            outputs = self.forward_model(pol_0, pol_45, pol_90, pol_135, depth, params)
            loss = self.training_loss(outputs)

            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.get("grad_clip", 1.0)
            )
            if not torch.isfinite(grad_norm):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.step()

            losses.append(loss.item())
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                water=params["water_type"],
                beta=f"{params['beta']:.1f}",
            )

        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_fused_psnr = 0.0
        total_air_psnr = 0.0
        count = 0
        total_cases = len(self.val_loader) * len(self.val_params)
        progress = tqdm(total=total_cases, desc=f"Validate {epoch + 1}")

        for batch in self.val_loader:
            pol_0, pol_45, pol_90, pol_135, _, depth = self.move_batch(batch)
            for params in self.val_params:
                outputs = self.forward_model(pol_0, pol_45, pol_90, pol_135, depth, params)
                _, _, fused_out, air_out, degraded_target, clean_target = outputs

                # Validation does not backpropagate. It only evaluates fused and restored outputs.
                val_loss = (
                    self.w_fused * charb(fused_out, degraded_target)
                    + self.w_air * charb(air_out, clean_target)
                )
                fused_mse = F.mse_loss(fused_out, degraded_target).clamp_min(1e-8)
                air_mse = F.mse_loss(air_out, clean_target).clamp_min(1e-8)

                total_loss += val_loss.item()
                total_fused_psnr += (10 * torch.log10(1 / fused_mse)).item()
                total_air_psnr += (10 * torch.log10(1 / air_mse)).item()
                count += 1
                progress.update(1)

        progress.close()
        if count == 0:
            raise RuntimeError("No validation samples were evaluated.")
        return {
            "loss": total_loss / count,
            "fused_psnr": total_fused_psnr / count,
            "air_psnr": total_air_psnr / count,
        }

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_psnr": self.best_val_psnr,
        }
        torch.save(checkpoint, self.checkpoint_dir / "latest.pth")
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pth")
        if (epoch + 1) % self.config.get("save_freq", 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch + 1:03d}.pth")

    def load_checkpoint(self, path):
        path = Path(path)
        if not path.exists():
            print(f"Checkpoint not found; start from epoch 1: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.best_val_psnr = checkpoint.get("best_val_psnr", -float("inf"))
        print(f"Resume from epoch {self.current_epoch}: {path}")

    def train(self):
        val_freq = max(1, self.config.get("val_freq", 5))
        for epoch in range(self.current_epoch, self.config["num_epochs"]):
            train_loss = self.train_one_epoch(epoch)
            run_validation = ((epoch + 1) % val_freq == 0
                              or epoch + 1 == self.config["num_epochs"])
            is_best = False

            if run_validation:
                val = self.validate(epoch)
                if val["fused_psnr"] > self.best_val_psnr:
                    self.best_val_psnr = val["fused_psnr"]
                    is_best = True
                print(
                    f"Epoch {epoch + 1:03d} | train={train_loss:.5f} | "
                    f"val={val['loss']:.5f} | fused={val['fused_psnr']:.3f} dB | "
                    f"air={val['air_psnr']:.3f} dB"
                    f"{' | best' if is_best else ''}"
                )
            else:
                print(f"Epoch {epoch + 1:03d} | train={train_loss:.5f}")

            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            self.save_checkpoint(epoch, is_best)


# ============================== Config =============================

def main():
    config = {
        "data_dir": "./data",
        "depth_mat_dir": "./data",
        "image_size": 512,
        "n_train": 90,
        "n_val": 15,
        "seed": 42,
        "rgb_base_ch": 32,
        "polar_base_ch": 32,
        "fusion_mid_ch": 32,
        "num_epochs": 200,
        "batch_size": 2,
        "val_batch_size": 1,
        "lr": 2e-4,
        "polar_lr": 2e-4,
        "fusion_lr": 2e-4,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
        "warmup_epochs": 3,
        "w_rgb": 0.4,
        "w_polar_img": 0.4,
        "w_fused": 1.0,
        "w_stokes": 0.2,
        "w_ssim": 0.2,
        "w_air": 0.3,
        "num_workers": 4,
        "output_dir": "./outputs",
        "save_freq": 10,
        "val_freq": 5,
        "val_include_identity": True,
        # Use None to train from the beginning.
        "resume": "./outputs/checkpoints/latest.pth",
    }

    trainer = Trainer(config)
    if config.get("resume"):
        trainer.load_checkpoint(config["resume"])
    trainer.train()


if __name__ == "__main__":
    main()
