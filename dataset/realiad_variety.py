"""
Real-IAD Variety Subset Dataset for AdaCLIP
This dataset is for zero-shot / few-shot anomaly detection prediction and evaluation.
"""
import os
import json
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


# All 12 categories in realiad_variety_subset
# Note: using the actual folder names (chip-inductor has hyphen)
REALIAD_VARIETY_CLS_NAMES = [
    'audio_jack_socket',
    'battery',
    'button_battery_holder',
    'chip-inductor',  # Using actual folder name with hyphen
    'connector',
    'flower_copper_shape',
    'gear',
    'hex_plug',
    'lock',
    'mouse_socket',
    'pencil_sharpener',
    'thyristor',
]


class RealIADVarietyDataset(data.Dataset):
    """
    Real-IAD Variety Subset Dataset for training/evaluation.
    
    Can use either:
    - realiad_variety_subset (original, anomaly="")
    - realiad_variety_subset_extracted (ground truth, anomaly=0/1)
    """
    def __init__(self, root, transform, target_transform, clsnames=None, training=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.training = training
        self.data_all = []
        
        if clsnames is None:
            self.cls_names = REALIAD_VARIETY_CLS_NAMES
        else:
            self.cls_names = clsnames
        
        # Load meta.json
        meta_path = os.path.join(root, 'meta.json')
        with open(meta_path, 'r') as f:
            self.meta_info = json.load(f)
        
        # Get test data
        test_data = self.meta_info.get('test', {})
        
        for cls_name in self.cls_names:
            # Try both the exact name and variations
            folder_name = cls_name
            if folder_name not in test_data:
                # Try with underscore replaced by hyphen
                folder_name = cls_name.replace('_', '-')
            if folder_name not in test_data:
                # Try with hyphen replaced by underscore
                folder_name = cls_name.replace('-', '_')
            
            if folder_name in test_data:
                for item in test_data[folder_name]:
                    item_copy = item.copy()
                    # Normalize cls_name (use the name from meta.json)
                    item_copy['cls_name_normalized'] = item.get('cls_name', cls_name)
                    self.data_all.append(item_copy)
        
        self.length = len(self.data_all)
        print(f"RealIAD Variety Dataset loaded: {self.length} samples from {root}")
    
    def __len__(self):
        return self.length
    
    def get_cls_names(self):
        return self.cls_names
    
    def __getitem__(self, index):
        data = self.data_all[index]
        img_path = os.path.join(self.root, data['img_path'])
        cls_name = data.get('cls_name_normalized', data.get('cls_name', 'unknown'))
        specie_name = data.get('specie_name', 'unknown')
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Handle anomaly value (can be int, string, or empty)
        anomaly_raw = data.get('anomaly', 0)
        if isinstance(anomaly_raw, str):
            if anomaly_raw == '' or anomaly_raw.lower() == 'null':
                anomaly = 0
            else:
                try:
                    anomaly = int(anomaly_raw)
                except ValueError:
                    anomaly = 0
        elif anomaly_raw is None:
            anomaly = 0
        else:
            anomaly = int(anomaly_raw)
        
        # Load mask if available
        mask_path = data.get('mask_path', '')
        if mask_path and mask_path != 'null' and mask_path is not None:
            full_mask_path = os.path.join(self.root, mask_path)
            if os.path.exists(full_mask_path):
                img_mask = np.array(Image.open(full_mask_path).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
            else:
                img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        else:
            img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
        
        # Apply transforms
        if self.transform is not None:
            img_transformed = self.transform(img)
        else:
            img_transformed = img
            
        if self.target_transform is not None:
            img_mask = self.target_transform(img_mask)
        
        return {
            'img': img_transformed,
            'img_mask': img_mask,
            'cls_name': cls_name,
            'anomaly': anomaly,
            'img_path': img_path,
            'specie_name': specie_name,
        }


class RealIADVarietyFewShotDataset(data.Dataset):
    """
    Real-IAD Variety Subset Dataset for few-shot learning.
    Provides normal reference images for each class.
    """
    def __init__(self, root, transform, clsnames=None, k_shot=8):
        self.root = root
        self.transform = transform
        self.k_shot = k_shot
        
        if clsnames is None:
            self.cls_names = REALIAD_VARIETY_CLS_NAMES
        else:
            self.cls_names = clsnames
        
        # Load meta.json
        meta_path = os.path.join(root, 'meta.json')
        with open(meta_path, 'r') as f:
            self.meta_info = json.load(f)
        
        # Get test data and filter OK samples for few-shot reference
        test_data = self.meta_info.get('test', {})
        
        self.reference_images = {}
        for cls_name in self.cls_names:
            folder_name = cls_name
            if folder_name not in test_data:
                folder_name = cls_name.replace('_', '-')
            if folder_name not in test_data:
                folder_name = cls_name.replace('-', '_')
            
            if folder_name in test_data:
                # Get OK samples (specie_name == 'OK')
                ok_samples = [item for item in test_data[folder_name] 
                             if item.get('specie_name', '') == 'OK']
                
                # Select up to k_shot samples
                self.reference_images[cls_name] = ok_samples[:k_shot]
        
        print(f"Few-shot reference loaded: {len(self.reference_images)} classes, up to {k_shot} shots each")
    
    def get_reference_images(self, cls_name, transform=None):
        """Get reference images for a specific class"""
        if cls_name not in self.reference_images:
            return []
        
        images = []
        for item in self.reference_images[cls_name]:
            img_path = os.path.join(self.root, item['img_path'])
            img = Image.open(img_path).convert('RGB')
            if transform is not None:
                img = transform(img)
            elif self.transform is not None:
                img = self.transform(img)
            images.append(img)
        
        return images
    
    def get_reference_paths(self, cls_name):
        """Get reference image paths for a specific class"""
        if cls_name not in self.reference_images:
            return []
        
        return [os.path.join(self.root, item['img_path']) 
                for item in self.reference_images[cls_name]]
