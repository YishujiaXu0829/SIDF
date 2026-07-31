
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from scipy.ndimage import uniform_filter
from torchvision import transforms
from tqdm import tqdm

from mosaic import PolarizationMosaicPipeline
from model import DualUCNN
from train_core_validation import (
    WATER_PARAMS,
    BETA_VALUES,
    IDENTITY_TYPE,
    degrade_batch,
    physics_restore,
    make_12ch,
    build_polar_input,
    load_depth_mat,
)


# ========================= Data and model =========================

IMAGE_RULES = [
    ("pol_0.png", "pol_45.png", "pol_90.png", "pol_135.png"),
    ("pol_0.jpg", "pol_45.jpg", "pol_90.jpg", "pol_135.jpg"),
    ("0.bmp", "45.bmp", "90.bmp", "135.bmp"),
    ("0.png", "45.png", "90.png", "135.png"),
]


def evaluation_params(include_identity=True):
    params = [
        {"water_type": water, "beta": beta, "t_is_one": False}
        for water in WATER_PARAMS
        for beta in BETA_VALUES
    ]
    if include_identity:
        params.append({"water_type": IDENTITY_TYPE, "beta": 0.0, "t_is_one": True})
    return params


def load_sample_names(data_dir, names_file=None):
    if names_file:
        with open(names_file, "r", encoding="utf-8") as file:
            names = json.load(file)
    else:
        names = sorted(path.name for path in Path(data_dir).iterdir() if path.is_dir())
    if not names:
        raise ValueError("No evaluation samples were found.")
    return names


def load_polarization_images(sample_dir, transform, device):
    sample_dir = Path(sample_dir)
    for filenames in IMAGE_RULES:
        paths = [sample_dir / filename for filename in filenames]
        if all(path.exists() for path in paths):
            return [
                transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                for path in paths
            ]
    raise FileNotFoundError(f"No complete polarization image group in {sample_dir}")


def load_depth(sample_name, depth_dir, device):
    if not depth_dir:
        return None
    depth_path = Path(depth_dir) / f"{sample_name}_depth.mat"
    if not depth_path.exists():
        return None
    depth = load_depth_mat(str(depth_path))
    if depth is None:
        return None
    return torch.from_numpy(depth)[None, None].to(device)


def load_model(config, device):
    model = DualUCNN(
        rgb_base_ch=config.get("rgb_base_ch", 32),
        polar_base_ch=config.get("polar_base_ch", 32),
        fusion_mid_ch=config.get("fusion_mid_ch", 32),
    ).to(device)

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ============================== Metrics ===========================

def split_12ch(tensor):
    return {
        "I0": torch.cat([tensor[:, 0:1], tensor[:, 4:5], tensor[:, 8:9]], dim=1),
        "I45": torch.cat([tensor[:, 1:2], tensor[:, 5:6], tensor[:, 9:10]], dim=1),
        "I90": torch.cat([tensor[:, 2:3], tensor[:, 6:7], tensor[:, 10:11]], dim=1),
        "I135": torch.cat([tensor[:, 3:4], tensor[:, 7:8], tensor[:, 11:12]], dim=1),
    }


def to_numpy(tensor):
    return tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float64)


def cpsnr(prediction, target):
    mse = np.mean((prediction - target) ** 2)
    return float(10 * np.log10(1.0 / (mse + 1e-8)))


def ssim_single(prediction, target, window=11):
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_p = uniform_filter(prediction, window)
    mu_t = uniform_filter(target, window)
    var_p = uniform_filter(prediction ** 2, window) - mu_p ** 2
    var_t = uniform_filter(target ** 2, window) - mu_t ** 2
    cov_pt = uniform_filter(prediction * target, window) - mu_p * mu_t
    numerator = (2 * mu_p * mu_t + c1) * (2 * cov_pt + c2)
    denominator = (mu_p ** 2 + mu_t ** 2 + c1) * (var_p + var_t + c2)
    return float(np.mean(numerator / (denominator + 1e-10)))


def ssim(prediction, target):
    return float(np.mean([
        ssim_single(prediction[:, :, channel], target[:, :, channel])
        for channel in range(prediction.shape[2])
    ]))


def stokes(polarizations):
    i0, i45 = polarizations["I0"], polarizations["I45"]
    i90, i135 = polarizations["I90"], polarizations["I135"]
    s0 = i0 + i90
    s1 = i0 - i90
    s2 = i45 - i135
    dolp = np.clip(np.sqrt(s1 ** 2 + s2 ** 2) / (s0 + 1e-10), 0, 1)
    aop = np.mod(0.5 * np.arctan2(s2, s1) * 180 / np.pi, 180)
    return {"S0": s0, "S1": s1, "S2": s2, "DoLP": dolp, "AoP": aop}


def evaluate(prediction_12ch, target_12ch):
    prediction = {key: to_numpy(value) for key, value in split_12ch(prediction_12ch).items()}
    target = {key: to_numpy(value) for key, value in split_12ch(target_12ch).items()}
    result = {}

    for name in ("I0", "I45", "I90", "I135"):
        result[f"{name}_CPSNR"] = cpsnr(prediction[name], target[name])
        result[f"{name}_SSIM"] = ssim(prediction[name], target[name])
    result["avg_CPSNR"] = float(np.mean([result[f"{name}_CPSNR"] for name in ("I0", "I45", "I90", "I135")]))
    result["avg_SSIM"] = float(np.mean([result[f"{name}_SSIM"] for name in ("I0", "I45", "I90", "I135")]))

    pred_stokes, target_stokes = stokes(prediction), stokes(target)
    for name in ("S0", "S1", "S2"):
        if name == "S0":
            pred_value, target_value = pred_stokes[name] / 2, target_stokes[name] / 2
        else:
            pred_value = (pred_stokes[name] + 1) / 2
            target_value = (target_stokes[name] + 1) / 2
        result[f"{name}_CPSNR"] = cpsnr(pred_value, target_value)
        result[f"{name}_SSIM"] = ssim(pred_value, target_value)

    result["DoLP_CPSNR"] = cpsnr(pred_stokes["DoLP"], target_stokes["DoLP"])
    result["DoLP_SSIM"] = ssim(pred_stokes["DoLP"], target_stokes["DoLP"])

    valid = (target_stokes["DoLP"] > 0.1) & (pred_stokes["DoLP"] > 0.1)
    difference = np.abs(pred_stokes["AoP"] - target_stokes["AoP"])
    angle_error = np.minimum(difference, 180 - difference)
    result["AoP_angle_error"] = float(np.mean(angle_error[valid])) if np.any(valid) else 0.0
    return result


# =========================== Evaluation loop ======================

def add_metrics(row, prefix, metrics):
    row.update({f"{prefix}_{key}": round(float(value), 6) for key, value in metrics.items()})


def write_results(path, rows):
    if not rows:
        raise RuntimeError("No evaluation results were produced.")

    fieldnames = list(rows[0])
    metric_fields = [key for key in fieldnames if key not in {"sample", "water_type", "beta"}]
    mean_row = {"sample": "MEAN", "water_type": "", "beta": ""}
    mean_row.update({
        key: round(float(np.mean([row[key] for row in rows])), 6)
        for key in metric_fields
    })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(mean_row)
        writer.writerows(rows)


def run_evaluation(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, device)
    pipeline = PolarizationMosaicPipeline()
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
    ])
    sample_names = load_sample_names(config["data_dir"], config.get("sample_names_file"))
    params_list = evaluation_params(config.get("include_identity", True))
    rows = []

    progress = tqdm(total=len(sample_names) * len(params_list), desc="Evaluate")
    with torch.no_grad():
        for sample_name in sample_names:
            sample_dir = Path(config["data_dir"]) / sample_name
            pol_0, pol_45, pol_90, pol_135 = load_polarization_images(
                sample_dir, transform, device,
            )
            clean_target = make_12ch(pol_0, pol_45, pol_90, pol_135).clamp(0, 1)
            depth = load_depth(sample_name, config.get("depth_mat_dir"), device)

            for params in params_list:
                d0, d45, d90, d135 = degrade_batch(
                    pol_0, pol_45, pol_90, pol_135, params, depth,
                )
                degraded_target = make_12ch(d0, d45, d90, d135).clamp(0, 1)
                _, mosaic_12ch, _ = pipeline(d0, d45, d90, d135)
                _, _, fused_out = model(mosaic_12ch, build_polar_input(mosaic_12ch))

                row = {
                    "sample": sample_name,
                    "water_type": params["water_type"],
                    "beta": params["beta"],
                }
                add_metrics(row, "uw", evaluate(fused_out, degraded_target))

                if config.get("run_restore", True):
                    air_out = physics_restore(fused_out.clamp(0, 1), params, depth)
                    add_metrics(row, "air", evaluate(air_out, clean_target))

                rows.append(row)
                progress.update(1)

    progress.close()
    write_results(config["output_csv"], rows)
    print(f"Device: {device}")
    print(f"Samples: {len(sample_names)}; conditions: {len(params_list)}")
    print(f"Results saved to: {config['output_csv']}")


def main():
    config = {
        "data_dir": "./data",
        "depth_mat_dir": "./data",
        "sample_names_file": "./outputs/val_sample_names.json",
        "checkpoint": "./outputs/checkpoints/best.pth",
        "output_csv": "./outputs/evaluation_results.csv",
        "image_size": 512,
        "rgb_base_ch": 32,
        "polar_base_ch": 32,
        "fusion_mid_ch": 32,
        "include_identity": True,
        "run_restore": True,
    }
    run_evaluation(config)


if __name__ == "__main__":
    main()
