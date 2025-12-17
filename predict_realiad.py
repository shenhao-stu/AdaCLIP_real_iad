import os
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.ndimage import gaussian_filter
import shutil

# Add project root to sys.path
import sys
sys.path.append(os.getcwd())

from method import AdaCLIP_Trainer
from tools import setup_seed

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def main():
    parser = argparse.ArgumentParser("AdaCLIP Prediction for Real-IAD", add_help=True)
    parser.add_argument("--input_root", type=str, default='/data/shenhao/AdaCLIP/data/Real-IAD/realiad_variety_subset')
    parser.add_argument("--output_root", type=str, default='/data/shenhao/AdaCLIP/data/Real-IAD/realiad_variety_subset_predict_zero_shot')
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_all.pth')
    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--prompting_depth", type=int, default=4)
    parser.add_argument("--prompting_length", type=int, default=5)
    parser.add_argument("--prompting_type", type=str, default='SD')
    parser.add_argument("--prompting_branch", type=str, default='VL')
    parser.add_argument("--use_hsf", type=str2bool, default=True)
    parser.add_argument("--k_clusters", type=int, default=20)
    
    args = parser.parse_args()
    
    setup_seed(111)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load config
    config_path = os.path.join('./model_configs', f'{args.model}.json')
    with open(config_path, 'r') as f:
        model_configs = json.load(f)
        
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]
    
    print(f"Loading model from {args.ckt_path}...")
    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=0.,
        device=device,
        image_size=args.image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)
    
    if os.path.isfile(args.ckt_path):
        model.load(args.ckt_path)
    else:
        print(f"Warning: Checkpoint {args.ckt_path} not found!")

    model.clip_model.eval()

    # Load Meta
    meta_path = os.path.join(args.input_root, 'meta.json')
    if not os.path.exists(meta_path):
        print(f"Meta file not found: {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    test_data = meta['test']
    new_test_data = {}

    # Create output root
    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)
        
    print(f"Processing data to {args.output_root}")

    for cls_name, items in test_data.items():
        new_items = []
        print(f"Processing class: {cls_name}, {len(items)} items")
        
        for item in tqdm(items):
            img_rel_path = item['img_path']
            src_img_path = os.path.join(args.input_root, img_rel_path)
            dst_img_path = os.path.join(args.output_root, img_rel_path)
            
            # Ensure directories exist
            os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
            
            # Symlink image
            if not os.path.exists(dst_img_path):
                if os.path.exists(src_img_path):
                    os.symlink(src_img_path, dst_img_path)
                else:
                    print(f"Warning: Source image missing {src_img_path}")
            
            specie = item['specie_name']
            
            if specie == 'OK':
                # Normal sample
                item['anomaly'] = "0"
                item['mask_path'] = ""
                new_items.append(item)
                continue
            
            # NG Sample - Predict
            if not os.path.exists(src_img_path):
                new_items.append(item)
                continue
                
            try:
                # Load and process image
                pil_img = Image.open(src_img_path).convert('RGB')
                orig_size = pil_img.size # (W, H) -> (1604, 1604)
                
                img_input = model.preprocess(pil_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    anomaly_map, anomaly_score = model.clip_model(img_input, [cls_name], aggregation=True)
                
                # Post-process
                # anomaly_map is Tensor [H, W] (already resized to args.image_size in model?? Check AdaCLIP.visual_text_similarity)
                # In visual_text_similarity, it interpolates to self.image_size.
                
                anomaly_map = anomaly_map.cpu().numpy() # Shape likely (1, 518, 518) or (518, 518)
                if len(anomaly_map.shape) == 3:
                    anomaly_map = anomaly_map[0] # (518, 518)
                
                anomaly_score = anomaly_score.cpu().numpy()
                if isinstance(anomaly_score, (list, np.ndarray)):
                    anomaly_score = float(anomaly_score[0]) if len(anomaly_score.shape) > 0 else float(anomaly_score)

                # Gaussian blur
                anomaly_map = gaussian_filter(anomaly_map, sigma=4)
                
                # Resize to original size (1604x1604)
                # cv2.resize takes (W, H)
                anomaly_map_resized = cv2.resize(anomaly_map, orig_size, interpolation=cv2.INTER_LINEAR)
                
                # Binarize
                # Threshold at 0.5 (assuming Softmax output from model)
                mask = (anomaly_map_resized > 0.5).astype(np.uint8) * 255
                
                # Save mask
                # format: ..._mask.png
                mask_rel_path = img_rel_path.replace('.png', '_mask.png')
                dst_mask_path = os.path.join(args.output_root, mask_rel_path)
                
                # Ensure mask directory exists (should be same as image)
                os.makedirs(os.path.dirname(dst_mask_path), exist_ok=True)
                
                cv2.imwrite(dst_mask_path, mask)
                
                # Update item
                item['anomaly'] = "1" # Force 1 for NG as per user "only predict NG images" (implies localization task)
                # If we wanted model prediction: item['anomaly'] = "1" if anomaly_score > 0.5 else "0"
                
                item['mask_path'] = mask_rel_path
                new_items.append(item)
                
            except Exception as e:
                print(f"Error processing {src_img_path}: {e}")
                new_items.append(item)

        new_test_data[cls_name] = new_items
        
    # Save meta.json
    output_meta_path = os.path.join(args.output_root, 'meta.json')
    with open(output_meta_path, 'w') as f:
        json.dump({"test": new_test_data}, f, indent=2)
    print(f"Saved results to {output_meta_path}")

if __name__ == "__main__":
    main()

