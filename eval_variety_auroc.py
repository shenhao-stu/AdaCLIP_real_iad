"""
Evaluation Script for Variety Subset
Computes I-AUROC, I-F1, P-AUROC, P-F1.
"""
import json
import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_f1_max(gt, pred):
    try:
        gt = np.array(gt)
        pred = np.array(pred)
        
        if len(np.unique(gt)) < 2:
            return 0.0
            
        precisions, recalls, _ = precision_recall_curve(gt, pred)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        f1_scores = f1_scores[np.isfinite(f1_scores)]
        
        if len(f1_scores) == 0:
            return 0.0
            
        return float(np.max(f1_scores))
    except Exception:
        return 0.0

def evaluate(gt_root, pred_root):
    gt_meta_path = os.path.join(gt_root, 'meta.json')
    pred_meta_path = os.path.join(pred_root, 'meta.json')
    
    if not os.path.exists(gt_meta_path):
        print(f"Error: GT meta file not found at {gt_meta_path}")
        return
    if not os.path.exists(pred_meta_path):
        print(f"Error: Pred meta file not found at {pred_meta_path}")
        return

    print(f"Loading GT from: {gt_meta_path}")
    print(f"Loading Pred from: {pred_meta_path}")

    gt_data = load_json(gt_meta_path)['test']
    pred_data = load_json(pred_meta_path)['test']
    
    # Categories
    categories = sorted(list(gt_data.keys()))
    
    print(f"\n{'Category':<25} {'I-AUROC':<10} {'I-F1':<10} {'P-AUROC':<10} {'P-F1':<10} {'Samples':<10}")
    print("-" * 85)

    img_aurocs = []
    img_f1s = []
    pix_aurocs = []
    pix_f1s = []
    
    for cat in categories:
        gt_samples = gt_data.get(cat, [])
        pred_samples = pred_data.get(cat, [])
        
        pred_map = {item['img_path']: item for item in pred_samples}
        
        img_gt_labels = []
        img_pred_scores = []
        pixel_gt_list = []
        pixel_pred_list = []
        
        for gt_item in gt_samples:
            img_path = gt_item['img_path']
            if img_path not in pred_map:
                continue
                
            pred_item = pred_map[img_path]
            
            # Image Level
            gt_anomaly = gt_item.get('anomaly', 0)
            if isinstance(gt_anomaly, str):
                gt_anomaly = int(gt_anomaly) if gt_anomaly else 0
            img_gt_labels.append(int(gt_anomaly))
            
            pred_anomaly = pred_item.get('anomaly', 0)
            if isinstance(pred_anomaly, str):
                pred_anomaly = float(pred_anomaly) if pred_anomaly else 0.0
            img_pred_scores.append(float(pred_anomaly))
            
            # Pixel Level
            gt_mask_path = gt_item.get('mask_path')
            pred_mask_path = pred_item.get('mask_path')
            
            # Load GT mask
            gt_mask = None
            if gt_mask_path:
                gt_mask_full = os.path.join(gt_root, gt_mask_path)
                if os.path.exists(gt_mask_full):
                    try:
                        m = Image.open(gt_mask_full).convert('L')
                        gt_mask = (np.array(m) > 128).astype(np.float32)
                    except:
                        gt_mask = None
            
            # Load Pred mask
            pred_mask = None
            if pred_mask_path:
                pred_mask_full = os.path.join(pred_root, pred_mask_path)
                if os.path.exists(pred_mask_full):
                    try:
                        m = Image.open(pred_mask_full).convert('L')
                        pred_mask = np.array(m).astype(np.float32) / 255.0
                    except:
                        pred_mask = None
            
            # Shape
            shape = None
            if gt_mask is not None:
                shape = gt_mask.shape
            elif pred_mask is not None:
                shape = pred_mask.shape
            else:
                img_full_path = os.path.join(gt_root, img_path)
                if os.path.exists(img_full_path):
                    try:
                        img = Image.open(img_full_path)
                        shape = (img.size[1], img.size[0])
                    except:
                        shape = None
            
            if shape is None:
                continue
                
            if gt_mask is None:
                gt_mask = np.zeros(shape, dtype=np.float32)
            if pred_mask is None:
                pred_mask = np.zeros(shape, dtype=np.float32)
                
            # Resize
            if pred_mask.shape != gt_mask.shape:
                try:
                    pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                    pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.BILINEAR)
                    pred_mask = np.array(pred_img).astype(np.float32) / 255.0
                except:
                    continue
                
            pixel_gt_list.append(gt_mask.flatten())
            pixel_pred_list.append(pred_mask.flatten())
            
        num_samples = len(img_gt_labels)
        if not img_gt_labels:
            print(f"{cat:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {0:<10}")
            continue
            
        # Metrics
        try:
            if len(set(img_gt_labels)) > 1:
                img_auroc = roc_auc_score(img_gt_labels, img_pred_scores)
            else:
                img_auroc = 0.5 
        except:
            img_auroc = 0.0
        
        img_f1 = compute_f1_max(img_gt_labels, img_pred_scores)
        
        try:
            if pixel_gt_list:
                all_gt = np.concatenate(pixel_gt_list)
                all_pred = np.concatenate(pixel_pred_list)
                
                if len(np.unique(all_gt)) > 1:
                    pixel_auroc = roc_auc_score(all_gt, all_pred)
                else:
                    pixel_auroc = 0.5
                
                pixel_f1 = compute_f1_max(all_gt, all_pred)
            else:
                pixel_auroc = 0.0
                pixel_f1 = 0.0
        except:
            pixel_auroc = 0.0
            pixel_f1 = 0.0
            
        img_aurocs.append(img_auroc)
        img_f1s.append(img_f1)
        pix_aurocs.append(pixel_auroc)
        pix_f1s.append(pixel_f1)
            
        print(f"{cat:<25} {img_auroc:.4f}     {img_f1:.4f}     {pixel_auroc:.4f}     {pixel_f1:.4f}     {num_samples:<10}")

    # Means
    mean_img_auroc = float(np.mean(img_aurocs)) if img_aurocs else float('nan')
    mean_img_f1 = float(np.mean(img_f1s)) if img_f1s else float('nan')
    mean_pix_auroc = float(np.mean(pix_aurocs)) if pix_aurocs else float('nan')
    mean_pix_f1 = float(np.mean(pix_f1s)) if pix_f1s else float('nan')

    print("-" * 85)
    print(f"Mean: I-AUROC={mean_img_auroc:.4f} I-F1={mean_img_f1:.4f} P-AUROC={mean_pix_auroc:.4f} P-F1={mean_pix_f1:.4f}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    GT_ROOT = os.path.join(current_dir, 'Variety')
    PRED_ROOT = os.path.join(current_dir, 'variety_test_set')
    evaluate(GT_ROOT, PRED_ROOT)
