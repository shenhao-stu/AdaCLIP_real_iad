"""
Zero-shot prediction on Real-IAD Variety Subset using AdaCLIP.

This script:
- Reads meta.json from the original Real-IAD variety subset.
- Runs AdaCLIP in a zero-shot manner to obtain image-level anomaly scores and
  pixel-level anomaly maps.
- Writes predictions into a new root folder (default:
  ./data/Real-IAD/realiad_variety_subset_predict_zero_shot) with:
    * Updated meta.json (anomaly -> "0"/"1", mask_path filled for NG samples).
    * Binary mask PNG files (0/255) with shape 1604x1604.
    * Mask files saved alongside their corresponding RGB images under the
      prediction root, e.g.:
      audio_jack_socket/NG/Deformation/S000011/S000011_C02_L05_W_..._mask.png

Normal (OK) images:
- anomaly is always set to "0".
- mask_path is left as an empty string "" and no mask file is written.

Abnormal (NG) images:
- anomaly is set to "1".
- mask_path points to the corresponding *_mask.png file.
"""

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataset import RealIADVarietyDataset
from method import AdaCLIP_Trainer


def load_model(ckt_path: str,
               model_name: str = "ViT-L-14-336",
               image_size: int = 518,
               prompting_depth: int = 4,
               prompting_length: int = 5,
               prompting_branch: str = "VL",
               prompting_type: str = "SD",
               use_hsf: bool = True,
               k_clusters: int = 20) -> AdaCLIP_Trainer:
    """Create and load AdaCLIP model for inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model config (same as in test.py)
    config_path = os.path.join("./model_configs", f"{model_name}.json")
    with open(config_path, "r") as f:
        model_configs = json.load(f)

    n_layers = model_configs["vision_cfg"]["layers"]
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=model_name,
        feat_list=features_list,
        input_dim=model_configs["vision_cfg"]["width"],
        output_dim=model_configs["embed_dim"],
        learning_rate=0.0,
        device=device,
        image_size=image_size,
        prompting_depth=prompting_depth,
        prompting_length=prompting_length,
        prompting_branch=prompting_branch,
        prompting_type=prompting_type,
        use_hsf=use_hsf,
        k_clusters=k_clusters,
    ).to(device)

    if not os.path.isfile(ckt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckt_path}")

    model.load(ckt_path)
    model.eval()
    return model


def is_ok_image(rel_img_path: str) -> bool:
    """Determine whether an image is a normal (OK) sample from its relative path."""
    # Convention in Real-IAD: ".../OK/..." vs ".../NG/DefectType/..."
    parts = rel_img_path.replace("\\", "/").split("/")
    return "OK" in parts


def build_mask_path(rel_img_path: str) -> str:
    """
    Given an image relative path like:
        audio_jack_socket/NG/Deformation/S000011/S000011_C02_L05_W_...png
    return the corresponding mask path:
        audio_jack_socket/NG/Deformation/S000011/S000011_C02_L05_W_..._mask.png
    """
    rel_img_path = rel_img_path.replace("\\", "/")
    base, ext = os.path.splitext(rel_img_path)
    return f"{base}_mask.png"


def resize_to_image(anomaly_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize anomaly map (H', W') to (target_h, target_w) using bilinear interpolation.
    """
    # Map is assumed to be float32, arbitrary range.
    # Normalize to [0, 255] for PIL, then back to [0, 1] after resizing.
    if anomaly_map.max() > 0:
        norm_map = anomaly_map / anomaly_map.max()
    else:
        norm_map = anomaly_map.copy()

    norm_map = (norm_map * 255.0).astype(np.uint8)
    pil_map = Image.fromarray(norm_map, mode="L")
    pil_map = pil_map.resize((target_w, target_h), resample=Image.BILINEAR)
    resized = np.array(pil_map).astype(np.float32) / 255.0
    return resized


def binarize_mask(anomaly_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarize anomaly map to 0/255 mask.
    """
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    return mask


def predict_and_save(
    data_root: str,
    pred_root: str,
    ckt_path: str,
    model_name: str = "ViT-L-14-336",
    image_size: int = 518,
    batch_size: int = 1,
    threshold: float = 0.5,
) -> None:
    """
    Main prediction pipeline:
    - Load dataset from data_root (original realiad_variety_subset).
    - Run AdaCLIP in zero-shot manner.
    - Save binary masks and updated meta.json to pred_root.
    """
    os.makedirs(pred_root, exist_ok=True)

    # Load original meta.json (we will copy and modify it)
    meta_path = os.path.join(data_root, "meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Load model
    model = load_model(
        ckt_path=ckt_path,
        model_name=model_name,
        image_size=image_size,
    )

    # Build dataset and dataloader
    # Use model.preprocess for images and model.transform for masks so that
    # DataLoader receives only tensors / simple types (no PIL.Image in batch).
    dataset = RealIADVarietyDataset(
        root=data_root,
        transform=model.preprocess,
        target_transform=model.transform,
        clsnames=None,
        training=False,
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    device = model.device

    # Map from relative img_path -> (anomaly_label_str, mask_rel_path or "")
    pred_info: Dict[str, Dict[str, Any]] = {}

    # Iterate over all samples
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in dataloader:
            # batch['img_path'] is absolute path from the dataset __getitem__
            img_paths_abs = batch["img_path"]
            cls_names = batch["cls_name"]

            imgs = batch["img"].to(device)

            # Run model (aggregation=True to get pixel map and image score)
            anomaly_maps, anomaly_scores = model.clip_model(imgs, cls_names, aggregation=True)

            # anomaly_maps: (B, H', W'), anomaly_scores: (B,)
            anomaly_maps = anomaly_maps.cpu().numpy()
            anomaly_scores = anomaly_scores.cpu().numpy()

            for idx_in_batch, abs_path in enumerate(img_paths_abs):
                abs_path = abs_path if isinstance(abs_path, str) else abs_path
                # Build relative path under data_root (for meta.json consistency)
                rel_path = os.path.relpath(abs_path, data_root)
                rel_path = rel_path.replace("\\", "/")

                # Decide whether this is OK or NG by path convention
                ok_flag = is_ok_image(rel_path)

                # Load original image to get its size (should be 1604x1604)
                with Image.open(abs_path).convert("RGB") as img_pil:
                    w, h = img_pil.size

                # Resize anomaly map to original size
                amap_small = anomaly_maps[idx_in_batch]
                amap_resized = resize_to_image(amap_small, target_h=h, target_w=w)

                # For OK images: anomaly="0", no mask written or referenced.
                # For NG images: anomaly="1", save binary mask.
                if ok_flag:
                    anomaly_label = "0"
                    mask_rel_path = ""
                else:
                    anomaly_label = "1"
                    # Binarize mask and save as PNG (0/255)
                    mask_bin = binarize_mask(amap_resized, threshold=threshold)
                    mask_pil = Image.fromarray(mask_bin.astype(np.uint8), mode="L")

                    mask_rel_path = build_mask_path(rel_path)
                    save_full_dir = os.path.join(pred_root, os.path.dirname(mask_rel_path))
                    os.makedirs(save_full_dir, exist_ok=True)
                    save_full_path = os.path.join(pred_root, mask_rel_path)
                    mask_pil.save(save_full_path)

                pred_info[rel_path] = {
                    "anomaly": anomaly_label,
                    "mask_path": mask_rel_path,
                    "score": float(anomaly_scores[idx_in_batch]),
                }

    # Optionally, copy RGB images from data_root to pred_root to ensure that
    # masks and images reside in the same folders as required.
    # Here we only ensure folders exist for masks; images are not mandatory
    # for evaluation, since evaluate_realiad.py reads RGBs from gt_root.

    # Build new meta.json under pred_root based on original meta.json order
    pred_meta = {"test": {}}
    original_test = meta.get("test", {})

    for cls_name, items in original_test.items():
        new_items = []
        for item in items:
            rel_path = item["img_path"].replace("\\", "/")
            # If we have prediction info, update anomaly and mask_path,
            # otherwise default to normal with empty mask.
            if rel_path in pred_info:
                info = pred_info[rel_path]
                new_item = dict(item)
                new_item["anomaly"] = str(int(info["anomaly"]))
                new_item["mask_path"] = info["mask_path"]
            else:
                new_item = dict(item)
                if is_ok_image(rel_path):
                    new_item["anomaly"] = "0"
                    new_item["mask_path"] = ""
                else:
                    # If NG but no prediction, mark as anomaly with empty mask.
                    new_item["anomaly"] = "1"
                    new_item["mask_path"] = ""
            new_items.append(new_item)
        pred_meta["test"][cls_name] = new_items

    pred_meta_path = os.path.join(pred_root, "meta.json")
    with open(pred_meta_path, "w", encoding="utf-8") as f:
        json.dump(pred_meta, f, indent=2, ensure_ascii=False)

    print(f"Prediction completed. Results saved to: {pred_meta_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Real-IAD Variety Zero-shot Prediction", add_help=True)

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/Real-IAD/realiad_variety_subset",
        help="Root path of the original Real-IAD variety subset.",
    )
    parser.add_argument(
        "--pred_root",
        type=str,
        default="./data/Real-IAD/realiad_variety_subset_predict_zero_shot",
        help="Root path to save prediction masks and meta.json.",
    )
    parser.add_argument(
        "--ckt_path",
        type=str,
        default="weights/pretrained_all.pth",
        help="Path to the AdaCLIP checkpoint for zero-shot inference.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14-336",
        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
        help="CLIP backbone to use.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=518,
        help="Input image size for AdaCLIP backbone.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (keep 1 to match existing trainer code).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold on anomaly map (after normalization) to binarize masks.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.batch_size != 1:
        raise NotImplementedError(
            "Currently, only batch size of 1 is supported due to unresolved bugs. "
            "Please set --batch_size to 1."
        )

    predict_and_save(
        data_root=args.data_root,
        pred_root=args.pred_root,
        ckt_path=args.ckt_path,
        model_name=args.model,
        image_size=args.image_size,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )


