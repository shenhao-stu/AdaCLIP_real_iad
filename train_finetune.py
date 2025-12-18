import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import torch
import sys

# Add project root to sys.path
sys.path.append(os.getcwd())

from tools import write2csv, setup_paths, setup_seed, log_metrics, Logger
from dataset import get_data
from method import AdaCLIP_Trainer

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def train(args):
    setup_seed(42)

    # Configurations
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_fig = args.save_fig

    # Set up paths
    model_name, image_dir, csv_path, log_path, ckp_path, tensorboard_logger = setup_paths(args)
    # Logger
    logger = Logger(log_path)

    # Print basic information
    for key, value in sorted(vars(args).items()):
        logger.info(f'{key} = {value}')

    logger.info('Model name: {:}'.format(model_name))

    config_path = os.path.join('./model_configs', f'{args.model}.json')

    # Prepare model
    with open(config_path, 'r') as f:
        model_configs = json.load(f)

    # Set up the feature hierarchy
    n_layers = model_configs['vision_cfg']['layers']
    substage = n_layers // 4
    features_list = [substage, substage * 2, substage * 3, substage * 4]

    model = AdaCLIP_Trainer(
        backbone=args.model,
        feat_list=features_list,
        input_dim=model_configs['vision_cfg']['width'],
        output_dim=model_configs['embed_dim'],
        learning_rate=learning_rate,
        device=device,
        image_size=image_size,
        prompting_depth=args.prompting_depth,
        prompting_length=args.prompting_length,
        prompting_branch=args.prompting_branch,
        prompting_type=args.prompting_type,
        use_hsf=args.use_hsf,
        k_clusters=args.k_clusters
    ).to(device)

    # Load Pretrained Weights
    if args.ckt_path and os.path.exists(args.ckt_path):
        logger.info(f"Loading pretrained weights from {args.ckt_path}")
        model.load(args.ckt_path)
    else:
        logger.info("No pretrained weights found or provided. Starting from scratch.")

    # Get Data
    # Training Data: realiad_256 (passed via args)
    train_data_cls_names, train_data, train_data_root = get_data(
        dataset_type_list=args.training_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=True)

    # Testing Data: realiad_256 (or others)
    test_data_cls_names, test_data, test_data_root = get_data(
        dataset_type_list=args.testing_data,
        transform=model.preprocess,
        target_transform=model.transform,
        training=False)
        
    # Subsample test data if requested
    if args.test_sample_ratio < 1.0:
        total_test = len(test_data)
        indices = torch.randperm(total_test)[:int(total_test * args.test_sample_ratio)]
        test_data = torch.utils.data.Subset(test_data, indices)
        logger.info(f"Subsampled test data: {len(test_data)}/{total_test} ({args.test_sample_ratio*100:.1f}%)")

    logger.info('Data Root: training, {:}; testing, {:}'.format(train_data_root, test_data_root))
    logger.info(f'Training samples: {len(train_data)}')
    logger.info(f'Testing samples: {len(test_data)}')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    best_f1 = -1e1

    for epoch in tqdm(range(epochs)):
        loss = model.train_epoch(train_dataloader)

        # Logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, loss))
            tensorboard_logger.add_scalar('loss', loss, epoch)

        # Validation
        if (epoch + 1) % args.valid_freq == 0 or (epoch == epochs - 1):
            if epoch == epochs - 1:
                save_fig_flag = save_fig
            else:
                save_fig_flag = False

            logger.info('=============================Testing ====================================')
            metric_dict = model.evaluation(
                test_dataloader,
                test_data_cls_names,
                save_fig_flag,
                image_dir,
            )

            log_metrics(
                metric_dict,
                logger,
                tensorboard_logger,
                epoch
            )

            # Check if 'Average' key exists, otherwise assume single dataset
            if 'Average' in metric_dict:
                f1_px = metric_dict['Average']['f1_px']
            else:
                # If only one class or dataset without average
                # This depends on how evaluation returns dict. 
                # Assuming 'Average' is always there if standard evaluation is used.
                f1_px = 0
                for k in metric_dict:
                    if isinstance(metric_dict[k], dict) and 'f1_px' in metric_dict[k]:
                        f1_px += metric_dict[k]['f1_px']
                f1_px /= len(metric_dict)

            # Save best
            if f1_px > best_f1:
                for k in metric_dict.keys():
                    write2csv(metric_dict[k], test_data_cls_names, k, csv_path)

                ckp_path_best = ckp_path + '_finetuned_best.pth'
                model.save(ckp_path_best)
                best_f1 = f1_px
                logger.info(f"New best model saved to {ckp_path_best} (F1: {best_f1:.4f})")
        
        # Periodic Save
        if (epoch + 1) % args.save_freq == 0:
            ckp_path_epoch = ckp_path + f'_epoch_{epoch+1}.pth'
            model.save(ckp_path_epoch)
            logger.info(f"Model saved to {ckp_path_epoch}")
    
    # Save final model
    ckp_path_final = ckp_path + '_finetuned_final.pth'
    model.save(ckp_path_final)
    logger.info(f"Final model saved to {ckp_path_final}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("AdaCLIP Fine-tuning", add_help=True)

    # Paths and configurations
    parser.add_argument("--training_data", type=str, default=["realiad_256"], nargs='+',
                        help="Datasets for training")
    parser.add_argument("--testing_data", type=str, default="realiad_256", 
                        help="Dataset for testing/validation")

    parser.add_argument("--save_path", type=str, default='./workspaces_finetune',
                        help="Directory to save results")

    parser.add_argument("--model", type=str, default="ViT-L-14-336",
                        choices=["ViT-B-16", "ViT-B-32", "ViT-L-14", "ViT-L-14-336"],
                        help="The CLIP model to be used")

    parser.add_argument("--save_fig", type=str2bool, default=False,
                        help="Save figures for visualizations")
    parser.add_argument("--ckt_path", type=str, default='weights/pretrained_all.pth', 
                        help="Path to the pre-trained model to fine-tune")

    # Hyper-parameters
    parser.add_argument("--exp_indx", type=int, default=1, help="Index of the experiment")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate (lower for finetuning)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--image_size", type=int, default=518, help="Size of the input images")
    parser.add_argument("--print_freq", type=int, default=1, help="Frequency of print statements")
    parser.add_argument("--valid_freq", typen=int, default=1, help="Frequency of validation")
    parser.add_argument("--save_freq", type=int, default=1, help="Frequency of saving checkpoint")
    parser.add_argument("--test_sample_ratio", type=float, default=1.0, help="Ratio of test data to use (0.0 to 1.0)")
    
    # Prompting parameters
    parser.add_argument("--prompting_depth", type=int, default=4, help="Depth of prompting")
    parser.add_argument("--prompting_length", type=int, default=5, help="Length of prompting")
    parser.add_argument("--prompting_type", type=str, default='SD', choices=['', 'S', 'D', 'SD'])
    parser.add_argument("--prompting_branch", type=str, default='VL', choices=['', 'V', 'L', 'VL'])

    parser.add_argument("--use_hsf", type=str2bool, default=True,
                        help="Use HSF for aggregation")
    parser.add_argument("--k_clusters", type=int, default=20, help="Number of clusters")

    args = parser.parse_args()

    if args.batch_size != 1:
        print("Warning: Batch size != 1 might have issues due to unresolved bugs in original implementation.")
        print("Proceeding with caution.")
    
    train(args)

