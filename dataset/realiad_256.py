"""
Real-IAD 256 Dataset for AdaCLIP Training
This dataset uses the realiad_256 images with directory scanning.
"""
import os
import json
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import glob

# Default root path (can be overridden)
REALIAD_256_ROOT = './data/Real-IAD'

# Placeholder, will be populated dynamically if not provided
REALIAD_256_CLS_NAMES = [] 

class RealIAD256Dataset(data.Dataset):
    """
    Real-IAD 256 Dataset for training AdaCLIP
    
    Scans directory structure:
    - root/realiad_256/[category]/OK/[sample]/images.jpg
    - root/realiad_256/[category]/NG/[defect_type]/[sample]/images.jpg (+ .png mask)
    """
    def __init__(self, clsnames=None, transform=None, target_transform=None, 
                 root=REALIAD_256_ROOT, aug_rate=0.2, training=True):
        self.root = root
        self.image_dir = os.path.join(root, 'realiad_256')
        self.transform = transform
        self.target_transform = target_transform
        self.aug_rate = aug_rate
        self.training = training
        self.data_all = []
        
        # Populate class names if not provided
        if clsnames is None or len(clsnames) == 0:
            if os.path.exists(self.image_dir):
                self.cls_names = [d for d in os.listdir(self.image_dir) 
                                 if os.path.isdir(os.path.join(self.image_dir, d))]
                self.cls_names.sort()
            else:
                self.cls_names = []
                print(f"Warning: Image directory {self.image_dir} not found.")
        else:
            self.cls_names = clsnames
            
        print(f"Loading RealIAD 256 data from {self.image_dir} for classes: {len(self.cls_names)}")
        
        for cls_name in self.cls_names:
            cls_dir = os.path.join(self.image_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
                
            # Process OK samples
            ok_dir = os.path.join(cls_dir, 'OK')
            if os.path.exists(ok_dir):
                for sample_id in os.listdir(ok_dir):
                    sample_path = os.path.join(ok_dir, sample_id)
                    if not os.path.isdir(sample_path): continue
                    
                    # Find jpg images
                    images = glob.glob(os.path.join(sample_path, "*.jpg"))
                    for img_path in images:
                        # Use relative path from root/realiad_256 if needed, or absolute
                        # Current code expects 'img_path' relative to image_dir in some places?
                        # Let's verify __getitem__. It uses: os.path.join(self.image_dir, data['img_path'])
                        # So we should store relative path.
                        
                        rel_path = os.path.relpath(img_path, self.image_dir)
                        
                        self.data_all.append({
                            'img_path': rel_path,
                            'mask_path': None,
                            'cls_name': cls_name,
                            'specie_name': 'OK',
                            'anomaly': 0
                        })

            # Process NG samples
            ng_dir = os.path.join(cls_dir, 'NG')
            if os.path.exists(ng_dir):
                for defect_type in os.listdir(ng_dir):
                    defect_path = os.path.join(ng_dir, defect_type)
                    if not os.path.isdir(defect_path): continue
                    
                    for sample_id in os.listdir(defect_path):
                        sample_path = os.path.join(defect_path, sample_id)
                        if not os.path.isdir(sample_path): continue
                        
                        images = glob.glob(os.path.join(sample_path, "*.jpg"))
                        for img_path in images:
                            rel_img_path = os.path.relpath(img_path, self.image_dir)
                            
                            # Assume mask has same basename but .png
                            mask_full_path = img_path.replace('.jpg', '.png')
                            rel_mask_path = None
                            
                            if os.path.exists(mask_full_path):
                                rel_mask_path = os.path.relpath(mask_full_path, self.image_dir)
                            
                            self.data_all.append({
                                'img_path': rel_img_path,
                                'mask_path': rel_mask_path,
                                'cls_name': cls_name,
                                'specie_name': defect_type,
                                'anomaly': 1
                            })
        
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
        # Filter samples for this class
        cls_samples = [d for d in self.data_all if d['cls_name'] == cls_name]
        if not cls_samples:
            return None, None
            
        img_info = random.sample(cls_samples, min(4, len(cls_samples)))
        
        img_ls = []
        mask_ls = []
        
        for data in img_info:
            img_path = os.path.join(self.image_dir, data['img_path'])
            
            try:
                img = Image.open(img_path).convert('RGB')
                img_ls.append(img)
                
                if data['anomaly'] == 0 or not data['mask_path']:
                    img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                else:
                    mask_path = os.path.join(self.image_dir, data['mask_path'])
                    if os.path.exists(mask_path):
                        img_mask = np.array(Image.open(mask_path).convert('L')) > 0
                        img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
                    else:
                        img_mask = Image.fromarray(np.zeros((img.size[1], img.size[0])), mode='L')
                
                mask_ls.append(img_mask)
            except Exception as e:
                print(f"Error loading image/mask: {e}")
                return None, None
        
        if not img_ls: return None, None

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
        
        img = None
        img_mask = None

        if self.training and random_number < self.aug_rate:
            img, img_mask = self.combine_img(cls_name)
        
        if img is None:
            # Fallback to regular loading
            try:
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
            except Exception as e:
                # Handle corrupt images?
                print(f"Error loading {img_path}: {e}")
                # Create dummy
                img = Image.new('RGB', (518, 518))
                img_mask = Image.new('L', (518, 518))

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
