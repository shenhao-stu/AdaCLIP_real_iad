import itertools
import subprocess
import os
import json
import datetime
import sys

# Configuration Grid
CONFIG = {
    "k_clusters": [10, 20, 40],
    "prompting_type": ["SD"],
    "image_size": [518],
    "model": ["ViT-L-14-336"],
    "ckt_path": [
        "weights/pretrained_all.pth",
        # "weights/pretrained_visa_clinicdb.pth",
        # "weights/pretrained_mvtec_colondb.pth"
    ]
}

# Paths
GT_ROOT = "/data/shenhao/AdaCLIP/data/Real-IAD/realiad_variety_subset_extracted"
INPUT_ROOT = "/data/shenhao/AdaCLIP/data/Real-IAD/realiad_variety_subset"
EXP_DIR = "exp"
RESULTS_FILE = "experiments_results.jsonl"
CONDA_EXEC = "/data/share/miniconda3/bin/conda"
ENV_NAME = "AdaCLIP"

def main():
    if not os.path.exists(EXP_DIR):
        os.makedirs(EXP_DIR)

    keys = CONFIG.keys()
    values = CONFIG.values()
    combinations = list(itertools.product(*values))

    print(f"Total experiments planned: {len(combinations)}")
    print(f"Results will be appended to: {RESULTS_FILE}")

    # --- Phase 1: Prediction ---
    print("\n" + "="*50)
    print("PHASE 1: PREDICTION (All configurations)")
    print("="*50)
    
    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        # Create a unique experiment name
        ckpt_name = os.path.basename(params['ckt_path']).replace('.pth', '').replace('pretrained_', '')
        exp_name = f"{params['model']}_k{params['k_clusters']}_{params['prompting_type']}_{ckpt_name}"
        output_root = os.path.join(EXP_DIR, exp_name)
        
        print(f"\n[{i+1}/{len(combinations)}] Prediction: {exp_name}")
        
        meta_json_path = os.path.join(output_root, 'meta.json')
        if os.path.exists(meta_json_path):
             print(f"  Prediction already exists at {output_root}, skipping.")
             continue

        print(f"  Running prediction...")
        cmd_predict = [
            CONDA_EXEC, "run", "-n", ENV_NAME, "python", "predict_zero_shot.py",
            "--input_root", INPUT_ROOT,
            "--output_root", output_root,
            "--ckt_path", params['ckt_path'],
            "--model", params['model'],
            "--image_size", str(params['image_size']),
            "--prompting_type", params['prompting_type'],
            "--k_clusters", str(params['k_clusters'])
        ]
        
        if not os.path.exists(output_root):
                os.makedirs(output_root)
        
        try:
            # Run with output captured to log
            with open(os.path.join(output_root, "predict.log"), "w") as log_file:
                    subprocess.run(cmd_predict, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"  Error running prediction for {exp_name}: {e}")
            continue

    # --- Phase 2: Evaluation ---
    print("\n" + "="*50)
    print("PHASE 2: EVALUATION (All configurations)")
    print("="*50)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        
        ckpt_name = os.path.basename(params['ckt_path']).replace('.pth', '').replace('pretrained_', '')
        exp_name = f"{params['model']}_k{params['k_clusters']}_{params['prompting_type']}_{ckpt_name}"
        output_root = os.path.join(EXP_DIR, exp_name)
        
        print(f"\n[{i+1}/{len(combinations)}] Evaluation: {exp_name}")

        eval_output_file = os.path.join(output_root, "evaluation_results.json")
        if os.path.exists(eval_output_file):
            print(f"  Evaluation already exists, reading results...")
        else:
            if not os.path.exists(os.path.join(output_root, 'meta.json')):
                print(f"  Skipping evaluation: Prediction missing for {exp_name}")
                continue

            print(f"  Running evaluation...")
            cmd_eval = [
                CONDA_EXEC, "run", "-n", ENV_NAME, "python", "evaluate_realiad.py",
                "--gt_root", GT_ROOT,
                "--pred_root", output_root,
                "--output_file", eval_output_file
            ]
            
            try:
                subprocess.run(cmd_eval, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                print(f"  Error running evaluation for {exp_name}: {e}")
                continue
            
        # Save Results
        if os.path.exists(eval_output_file):
            try:
                with open(eval_output_file, 'r') as f:
                    eval_data = json.load(f)
                
                result_record = {
                    "experiment_name": exp_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "parameters": params,
                    "metrics": {
                        "mean_image_auroc": eval_data.get('mean_image_auroc'),
                        "mean_image_f1": eval_data.get('mean_image_f1'),
                        "mean_pixel_auroc": eval_data.get('mean_pixel_auroc'),
                        "mean_pixel_f1": eval_data.get('mean_pixel_f1'),
                        "class_results": eval_data.get('class_results', [])
                    },
                    "output_dir": output_root
                }
                
                # Check if result already in jsonl to avoid duplicates (optional but good)
                # For simplicity, we just append. User can dedupe later.
                with open(RESULTS_FILE, 'a') as f:
                    f.write(json.dumps(result_record) + "\n")
                    
                print(f"  Saved metrics: I-AUROC={result_record['metrics']['mean_image_auroc']:.4f}, I-F1={result_record['metrics'].get('mean_image_f1', 0):.4f}, P-AUROC={result_record['metrics']['mean_pixel_auroc']:.4f}, P-F1={result_record['metrics'].get('mean_pixel_f1', 0):.4f}")
            except Exception as e:
                print(f"  Error parsing evaluation results: {e}")
        else:
             print(f"  Evaluation output file not found.")

if __name__ == "__main__":
    main()
