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
# from config import (
#     DATASET_PATH, TASK_ID, TRAIN_VAL_TEST_SPLIT,
#     TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
# )


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

class NIFTISegmentationDatasetCT(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None, source = 'florida'):
        super(NIFTISegmentationDatasetCT, self).__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.source = source
        # Get all patient IDs
        self.patient_ids = [filename.replace(".nii", "") for filename in os.listdir(os.path.join(root_dir, "volumes"))]

        self.patient_ids = natsorted(self.patient_ids)
        random.seed(42)
        random.shuffle(self.patient_ids)
        
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
    
    def load_nifti_volume(self, path):
        # Load NIFTI file and return as a 3D volume
        nifti_data = nib.load(path)
        volume_3d = np.array(nifti_data.dataobj, dtype=np.float32)
        volume_3d = np.transpose(volume_3d, (2,0,1))
        return volume_3d
    
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
        
        img_path = os.path.join(self.root_dir, "volumes", patient_id + ".nii")
        label_path = os.path.join(self.root_dir, "segmentations", patient_id + ".nii")
        
        img_3d = self.load_nifti_volume(img_path)
        label_3d_indices = self.load_nifti_volume(label_path)
        label_3d = np.eye(3)[label_3d_indices.astype(np.uint8)]

        processed_out = {'name': patient_id, 'image': img_3d, 'label': label_3d}

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