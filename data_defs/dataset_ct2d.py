import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import random
import nibabel as nib
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torch.utils.data import DataLoader
from natsort import natsorted
import cv2
from data_defs.utils import create_gaussian_label_volume, create_circular_label_volume, create_approximated_spline_volume
from config import *


def seed_everything(seed_value = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True  # ensure deterministic behavior for cuDNN
        torch.backends.cudnn.benchmark = False  # it can be beneficial to turn this off as it can introduce randomness

    print(f'Set all random seeds to {seed_value}.')


class FrameDatasetCT(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None):
        super(FrameDatasetCT, self).__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.patient_ids = natsorted(os.listdir(os.path.join(root_dir, "images")))
        self.approximation = APPROXIMATION #choose from circular, spline
        self.approximation_error = APPROXIMATION_ERROR 
        self.approximation_percent = APPROXIMATION_PERCENT
        self.train_subset = TRAIN_SUBSET
        self.patient_ids = natsorted(self.patient_ids)
        random.seed(1)
        random.shuffle(self.patient_ids)
        self.num_available_samples = int(len(self.patient_ids)*self.train_subset)
        self.patient_ids = self.patient_ids[:self.num_available_samples] #ensure this line is not activated
        
        # Split the patient IDs into train, validation, and test sets
        num_patients = len(self.patient_ids)
        train_count, val_count, test_count = [int(ratio * num_patients) for ratio in split_ratios]
        self.train_ids = self.patient_ids[:train_count]
        self.val_ids = self.patient_ids[train_count:train_count+val_count]
        self.test_ids = self.patient_ids[train_count+val_count:]
    
    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train_ids)
        elif self.mode == "val":
            return len(self.val_ids)
        elif self.mode == "test":
            return len(self.test_ids)
        return len(self.patient_ids)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == "train":
            patient_id = self.train_ids[idx]
        elif self.mode == "val":
            patient_id = self.val_ids[idx]
        elif self.mode == "test":
            patient_id = self.test_ids[idx]
        else:
            patient_id = self.patient_ids[idx]
        
        img_path = os.path.join(self.root_dir, "images", patient_id)
        label_path = os.path.join(self.root_dir, "masks", patient_id)            
        
        img, label = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        if self.approximation == 'circular':
            random_state = np.random.random()*100.
            if random_state < self.approximation_percent:
                label_adjusted = create_circular_label_volume(label, self.approximation_error)
            else:
                label_adjusted = label
        elif self.approximation == 'gaussian':
            label_adjusted = create_gaussian_label_volume(label, self.approximation_error)
        elif self.approximation == 'spline':
            random_state = np.random.random()*100.
            if random_state < self.approximation_percent:
                label_adjusted = create_approximated_spline_volume(label)
            else:
                label_adjusted = label
        elif self.approximation is None:
            label_adjusted = label
        else:
            raise NotImplementedError

        spline_labels = np.zeros((N_MC_SAMPLES, label.shape[0], label.shape[1]))

        if LOSS_FN == 'DiscrepancyLoss':
            for i in range(N_MC_SAMPLES):
                spline_labels[i] = create_approximated_spline_volume(label, num_points = 5 + N_SPLINE_POINTS - i)

        img  = (img/255.).astype(np.float32) #important : check if this applies
        if label_adjusted.max() == 255.:
            label_adjusted  = (label_adjusted/255.).astype(np.float32)
            spline_labels  = (spline_labels/255.).astype(np.float32)
        else:
            label_adjusted = label_adjusted.astype(np.float32)
            spline_labels = spline_labels.astype(np.float32)

        processed_out = {'name': patient_id, 'image': img, 'label': label_adjusted, 'original_label': label, 'spline_labels' : spline_labels}

        if self.transforms:
            if self.mode == "train":
                processed_out = self.transforms[0](processed_out)
            elif self.mode == "val":
                processed_out = self.transforms[1](processed_out)
            elif self.mode == "test":
                processed_out = self.transforms[2](processed_out)
            else:
                processed_out = self.transforms(processed_out)
        
        return processed_out
    
class FrameDatasetCTClassification(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None):
        super(FrameDatasetCTClassification, self).__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.patient_ids = natsorted(os.listdir(os.path.join(root_dir, "images")))
        self.train_subset = TRAIN_SUBSET
        self.patient_ids = natsorted(self.patient_ids)
        random.seed(1)
        random.shuffle(self.patient_ids)
        self.num_available_samples = int(len(self.patient_ids)*self.train_subset)
        self.patient_ids = self.patient_ids[:self.num_available_samples] #ensure this line is not activated
        
        # Split the patient IDs into train, validation, and test sets
        num_patients = len(self.patient_ids)
        train_count, val_count, test_count = [int(ratio * num_patients) for ratio in split_ratios]
        self.train_ids = self.patient_ids[:train_count]
        self.val_ids = self.patient_ids[train_count:train_count+val_count]
        self.test_ids = self.patient_ids[train_count+val_count:]
    
    def set_mode(self, mode):
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return len(self.train_ids)
        elif self.mode == "val":
            return len(self.val_ids)
        elif self.mode == "test":
            return len(self.test_ids)
        return len(self.patient_ids)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == "train":
            patient_id = self.train_ids[idx]
        elif self.mode == "val":
            patient_id = self.val_ids[idx]
        elif self.mode == "test":
            patient_id = self.test_ids[idx]
        else:
            patient_id = self.patient_ids[idx]
        
        img_path = os.path.join(self.root_dir, "images", patient_id)
        label_path = os.path.join(self.root_dir, "masks", patient_id)            
        
        img, label = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img  = (img/255.).astype(np.float32) #important : check if this applies

        if label.sum() > 0.01*512*512:
            label_binary = [0.,1.]
        else:
            label_binary = [1.,0.]

        label = (label/255.).astype(np.float32)
        processed_out = {'name': patient_id, 'image': img, 'label': label_binary, 'original_label' : label}

        if self.transforms:
            if self.mode == "train":
                processed_out = self.transforms[0](processed_out)
            elif self.mode == "val":
                processed_out = self.transforms[1](processed_out)
            elif self.mode == "test":
                processed_out = self.transforms[2](processed_out)
            else:
                processed_out = self.transforms(processed_out)
        
        return processed_out
    

