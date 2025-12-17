import json
import os
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import sys

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    
    print(f"\n{'Category':<25} {'Image AUROC':<15} {'Pixel AUROC':<15} {'Samples':<10}")
    print("-" * 75)

    # 用于统计每个类别的 AUROC 以便最后取平均
    img_aurocs = []
    pix_aurocs = []
    
    for cat in categories:
        gt_samples = gt_data.get(cat, [])
        pred_samples = pred_data.get(cat, [])
        
        # Map pred samples by img_path for easy lookup
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
            
            # --- Image Level ---
            # GT: anomaly (0/1)
            img_gt_labels.append(int(gt_item['anomaly']))
            
            # Pred: anomaly (0/1 or score). 
            img_pred_scores.append(float(pred_item['anomaly']))
            
            # --- Pixel Level ---
            gt_mask_path = gt_item.get('mask_path')
            pred_mask_path = pred_item.get('mask_path')
            
            # Load GT mask (Binary)
            gt_mask = None
            if gt_mask_path:
                gt_mask_full = os.path.join(gt_root, gt_mask_path)
                if os.path.exists(gt_mask_full):
                    try:
                        m = Image.open(gt_mask_full).convert('L')
                        gt_mask = (np.array(m) > 128).astype(np.float32) # Binarize GT
                    except:
                        gt_mask = None
            
            # Load Pred mask (Continuous)
            pred_mask = None
            if pred_mask_path:
                pred_mask_full = os.path.join(pred_root, pred_mask_path)
                if os.path.exists(pred_mask_full):
                    try:
                        m = Image.open(pred_mask_full).convert('L')
                        pred_mask = np.array(m).astype(np.float32) / 255.0
                    except:
                        pred_mask = None
            
            # Determine shape
            shape = None
            if gt_mask is not None:
                shape = gt_mask.shape
            elif pred_mask is not None:
                shape = pred_mask.shape
            else:
                # Need to read original image to get shape
                img_full_path = os.path.join(gt_root, img_path)
                if os.path.exists(img_full_path):
                    try:
                        img = Image.open(img_full_path)
                        shape = (img.size[1], img.size[0]) # H, W
                    except:
                        shape = None
            
            if shape is None:
                continue
                
            # Fill None with zeros
            if gt_mask is None:
                gt_mask = np.zeros(shape, dtype=np.float32)
            if pred_mask is None:
                pred_mask = np.zeros(shape, dtype=np.float32)
                
            # Resize Pred to GT if needed
            if pred_mask.shape != gt_mask.shape:
                try:
                    pred_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                    pred_img = pred_img.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.BILINEAR)
                    pred_mask = np.array(pred_img).astype(np.float32) / 255.0
                except:
                    continue
                
            pixel_gt_list.append(gt_mask.flatten())
            pixel_pred_list.append(pred_mask.flatten())
            
        # Calculate Metrics
        num_samples = len(img_gt_labels)
        if not img_gt_labels:
            print(f"{cat:<25} {'N/A':<15} {'N/A':<15} {0:<10}")
            continue
            
        # Image AUROC
        try:
            if len(set(img_gt_labels)) > 1:
                img_auroc = roc_auc_score(img_gt_labels, img_pred_scores)
            else:
                img_auroc = 0.5 
        except Exception as e:
            img_auroc = 0.0
            
        # Pixel AUROC
        try:
            if pixel_gt_list:
                all_gt = np.concatenate(pixel_gt_list)
                all_pred = np.concatenate(pixel_pred_list)
                
                if len(np.unique(all_gt)) > 1:
                    pixel_auroc = roc_auc_score(all_gt, all_pred)
                else:
                    pixel_auroc = 0.5
            else:
                pixel_auroc = 0.0
        except Exception as e:
            pixel_auroc = 0.0

        # 累计有效类别的 AUROC
        img_aurocs.append(img_auroc)
        pix_aurocs.append(pixel_auroc)
            
        print(f"{cat:<25} {img_auroc:.4f}           {pixel_auroc:.4f}           {num_samples:<10}")

    # 所有类别的宏平均 AUROC
    if img_aurocs:
        mean_img_auroc = float(np.mean(img_aurocs))
    else:
        mean_img_auroc = float('nan')

    if pix_aurocs:
        mean_pix_auroc = float(np.mean(pix_aurocs))
    else:
        mean_pix_auroc = float('nan')

    print("-" * 75)
    print(f"Mean over classes: image_AUROC={mean_img_auroc:.4f}  pixel_AUROC={mean_pix_auroc:.4f}")

if __name__ == '__main__':
    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    GT_ROOT = os.path.join(current_dir, 'Variety')
    PRED_ROOT = os.path.join(current_dir, 'variety_test_set')
    
    evaluate(GT_ROOT, PRED_ROOT)
