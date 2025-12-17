"""
Real-IAD 256 Dataset for AdaCLIP Training
This dataset uses the realiad_256 images with realiad_jsons meta files.
"""
import os
import json
import random
import numpy as np
import torch.utils.data as data
from PIL import Image


# All 30 categories in Real-IAD 256
REALIAD_256_CLS_NAMES = [
    'audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser',
    'fire_hood', 'mint', 'mounts', 'pcb', 'phone_battery',
    'plastic_nut', 'plastic_plug', 'porcelain_doll', 'regulator', 'rolled_strip_base',
    'sim_card_set', 'switch', 'tape', 'terminalblock', 'toothbrush',
    'toy', 'toy_brick', 'transistor1', 'u_block', 'usb',
    'usb_adaptor', 'vcpill', 'wooden_beads', 'woodstick', 'zipper'
]

# Default root path (can be overridden)
REALIAD_256_ROOT = './data/Real-IAD'


class RealIAD256Dataset(data.Dataset):
    """
    Real-IAD 256 Dataset for training AdaCLIP
    
    Uses the JSON format from realiad_jsons folder.
    Directory structure expected:
    - root/realiad_256/[category]/OK/[sample]/images...
    - root/realiad_256/[category]/NG/[defect_type]/[sample]/images...
    - root/realiad_jsons/[category].json
    """
    def __init__(self, clsnames=None, transform=None, target_transform=None, 
                 root=REALIAD_256_ROOT, aug_rate=0.2, training=True):
        self.root = root
        self.image_dir = os.path.join(root, 'realiad_256')
        self.json_dir = os.path.join(root, 'realiad_jsons')
        self.transform = transform
        self.target_transform = target_transform
        self.aug_rate = aug_rate
        self.training = training
        self.data_all = []
        
        if clsnames is None:
            self.cls_names = REALIAD_256_CLS_NAMES
        else:
            self.cls_names = clsnames
        
        # Load data from JSON files
        self.meta_info = {'train': {}, 'test': {}}
        
        for cls_name in self.cls_names:
            json_path = os.path.join(self.json_dir, f'{cls_name}.json')
            if not os.path.exists(json_path):
                print(f"Warning: JSON file not found for {cls_name}, skipping...")
                continue
            
            with open(json_path, 'r') as f:
                info = json.load(f)
            
            meta = info['meta']
            prefix = meta['prefix']
            normal_class = meta['normal_class']
            
            # Process train data
            train_samples = []
            missing_count = 0
            for sample in info['train']:
                full_path = os.path.join(self.image_dir, prefix, sample['image_path'])
                if os.path.exists(full_path):
                    train_samples.append({
                        'img_path': os.path.join(prefix, sample['image_path']),
                        'mask_path': '',
                        'cls_name': sample['category'],
                        'specie_name': sample['anomaly_class'],
                        'anomaly': 0  # Training samples are all normal
                    })
                else:
                    missing_count += 1
            
            if missing_count > 0:
                print(f"[{cls_name}] Skipped {missing_count} missing training files")
            
            self.meta_info['train'][cls_name] = train_samples
            
            # Process test data
            test_samples = []
            missing_count = 0
            for sample in info['test']:
                full_path = os.path.join(self.image_dir, prefix, sample['image_path'])
                if not os.path.exists(full_path):
                    missing_count += 1
                    continue
                    
                is_normal = (sample['mask_path'] is None or 
                            sample['anomaly_class'] == normal_class)
                
                mask_path = ''
                if not is_normal and sample['mask_path']:
                    mask_path = os.path.join(prefix, sample['mask_path'])
                
                test_samples.append({
                    'img_path': os.path.join(prefix, sample['image_path']),
                    'mask_path': mask_path,
                    'cls_name': sample['category'],
                    'specie_name': sample['anomaly_class'],
                    'anomaly': 0 if is_normal else 1
                })
            
            if missing_count > 0:
                print(f"[{cls_name}] Skipped {missing_count} missing test files")
                
            self.meta_info['test'][cls_name] = test_samples
        
        # Use test data for both training and testing (following AdaCLIP convention)
        data_source = self.meta_info['test']
        for cls_name in self.cls_names:
            if cls_name in data_source:
                self.data_all.extend(data_source[cls_name])
        
        self.length = len(self.data_all)
        print(f"RealIAD 256 Dataset loaded: {self.length} samples, training={training}")
    
    def __len__(self):
        return self.length
    
    def get_cls_names(self):
        return self.cls_names
    
    def combine_img(self, cls_name):
        """
        Data augmentation: combine four images into one
        """
        if cls_name not in self.meta_info['test']:
            return None, None
        
        img_info = random.sample(self.meta_info['test'][cls_name], 
                                min(4, len(self.meta_info['test'][cls_name])))
        
        img_ls = []
        mask_ls = []
        
        for data in img_info:
            img_path = os.path.join(self.image_dir, data['img_path'])
            
            img = Image.open(img_path).convert('RGB')
            img_ls.append(img)
            
            if not data['anomaly'] or not data['mask_path']:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                mask_path = os.path.join(self.image_dir, data['mask_path'])
                if os.path.exists(mask_path):
                    img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                else:
                    img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            
            mask_ls.append(img_mask)
        
        # Combine images
        image_width, image_height = img_ls[0].size
        result_image = Image.new("RGB", (2 * image_width, 2 * image_height))
        for i, img in enumerate(img_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_image.paste(img.resize((image_width, image_height)), (x, y))
        
        # Combine masks
        result_mask = Image.new("L", (2 * image_width, 2 * image_height))
        for i, mask in enumerate(mask_ls):
            row = i // 2
            col = i % 2
            x = col * image_width
            y = row * image_height
            result_mask.paste(mask.resize((image_width, image_height)), (x, y))
        
        return result_image, result_mask
    
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = os.path.join(self.image_dir, data['img_path'])
        cls_name = data['cls_name']
        anomaly = data['anomaly']
        
        random_number = random.random()
        
        if self.training and random_number < self.aug_rate:
            img, img_mask = self.combine_img(cls_name)
            if img is None:
                # Fallback to regular loading
                img = Image.open(img_path).convert('RGB')
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            img = Image.open(img_path).convert('RGB')
            
            if anomaly == 0 or not data['mask_path']:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
            else:
                mask_path = os.path.join(self.image_dir, data['mask_path'])
                if os.path.exists(mask_path):
                    img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                    img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                else:
                    img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        
        # Apply transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None and img_mask is not None:
            img_mask = self.target_transform(img_mask)
        if img_mask is None:
            img_mask = []
        
        return {
            'img': img,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': img_path
        }
