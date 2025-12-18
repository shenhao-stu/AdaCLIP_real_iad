import json
import os
from typing import Dict, List

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score


def load_meta(meta_path: str) -> Dict[str, List[dict]]:
    with open(meta_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["test"]


def evaluate_f1(gt_root: str, pred_root: str):
    gt_meta_path = os.path.join(gt_root, 'meta.json')
    pred_meta_path = os.path.join(pred_root, 'meta.json')

    print(f"GT meta:   {gt_meta_path}")
    print(f"Pred meta: {pred_meta_path}")

    gt_meta = load_meta(gt_meta_path)
    pred_meta = load_meta(pred_meta_path)

    categories = sorted(set(gt_meta.keys()) & set(pred_meta.keys()))

    print(f"\n{'Category':<25} {'Img-F1':<10} {'Pix-F1':<10} {'Samples':<10}")
    print('-' * 70)

    img_f1_list = []
    pix_f1_list = []

    for cat in categories:
        gt_items = gt_meta[cat]
        pred_items = pred_meta[cat]

        pred_index = {item['img_path']: item for item in pred_items}

        y_true = []      # image-level GT
        y_pred = []      # image-level Pred

        pix_gt_all = []  # pixel-level GT (flattened)
        pix_pr_all = []  # pixel-level Pred (flattened)

        for gt_rec in gt_items:
            img_path = gt_rec['img_path']
            if img_path not in pred_index:
                continue
            pred_rec = pred_index[img_path]

            # ----- image-level -----
            y_true.append(int(gt_rec['anomaly']))
            y_pred.append(int(pred_rec['anomaly']))

            # ----- pixel-level -----
            gt_mask_path = gt_rec.get('mask_path')
            pred_mask_path = pred_rec.get('mask_path')

            # 读取 GT mask（二值，0/1）；若为 None 则全 0
            gt_mask = None
            if gt_mask_path is not None:
                m_path = os.path.join(gt_root, gt_mask_path)
                if os.path.exists(m_path):
                    try:
                        m = Image.open(m_path).convert('L')
                        gt_mask = (np.array(m) > 0).astype(np.uint8)
                    except Exception:
                        gt_mask = None

            # 读取预测 mask（二值化 >0 为 1）；若为 None 则全 0
            pr_mask = None
            if pred_mask_path is not None:
                m_path = os.path.join(pred_root, pred_mask_path)
                if os.path.exists(m_path):
                    try:
                        m = Image.open(m_path).convert('L')
                        pr_mask = (np.array(m) > 0).astype(np.uint8)
                    except Exception:
                        pr_mask = None

            # 决定共同 shape
            shape = None
            if gt_mask is not None:
                shape = gt_mask.shape
            elif pr_mask is not None:
                shape = pr_mask.shape
            else:
                # 都没有 mask，就跳过 pixel 级这张图
                continue

            # 补全 None 为全 0
            if gt_mask is None:
                gt_mask = np.zeros(shape, dtype=np.uint8)
            if pr_mask is None:
                pr_mask = np.zeros(shape, dtype=np.uint8)

            # 若尺寸不一致，resize 预测到 GT 尺寸
            if pr_mask.shape != gt_mask.shape:
                pr_img = Image.fromarray((pr_mask * 255).astype(np.uint8))
                pr_img = pr_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                pr_mask = (np.array(pr_img) > 0).astype(np.uint8)

            pix_gt_all.append(gt_mask.reshape(-1))
            pix_pr_all.append(pr_mask.reshape(-1))

        # ---- 计算 image-level F1 ----
        num_samples = len(y_true)
        if num_samples == 0:
            print(f"{cat:<25} {'N/A':<10} {'N/A':<10} {0:<10}")
            continue

        try:
            img_f1 = f1_score(y_true, y_pred, average='binary', pos_label=1)
        except Exception:
            img_f1 = float('nan')

        # ---- 计算 pixel-level F1 ----
        if pix_gt_all:
            gt_flat = np.concatenate(pix_gt_all)
            pr_flat = np.concatenate(pix_pr_all)
            try:
                pix_f1 = f1_score(gt_flat, pr_flat, average='binary', pos_label=1)
            except Exception:
                pix_f1 = float('nan')
        else:
            pix_f1 = float('nan')

        img_f1_list.append(img_f1)
        pix_f1_list.append(pix_f1)

        print(f"{cat:<25} {img_f1:.4f}    {pix_f1:.4f}    {num_samples:<10}")

    # 类间平均
    mean_img_f1 = float(np.nanmean(img_f1_list)) if img_f1_list else float('nan')
    mean_pix_f1 = float(np.nanmean(pix_f1_list)) if pix_f1_list else float('nan')

    print('-' * 70)
    print(f"Mean over classes: Img-F1={mean_img_f1:.4f}  Pix-F1={mean_pix_f1:.4f}")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_root = os.path.join(base_dir, 'Variety')
    pred_root = os.path.join(base_dir, '1_test')
    evaluate_f1(gt_root, pred_root)
