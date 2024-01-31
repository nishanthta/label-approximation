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
from config import (
    DATASET_PATH, TASK_ID, TRAIN_VAL_TEST_SPLIT,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE
)


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


class FrameDataset(Dataset):
    def __init__(self, root_dir, transforms=None, mode=None, split_ratios = [0.8,0.1,0.1]):
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        # Get all patient IDs
        self.frame_ids = natsorted(os.listdir(os.path.join(self.root_dir, 'frames')))
        self.split_ratios = split_ratios

        random.seed(42)
        random.shuffle(self.frame_ids)

        num_frames = len(self.frame_ids)
        train_count, val_count, test_count = [int(ratio * num_frames) for ratio in split_ratios]
        self.train_ids = self.frame_ids[:train_count]
        self.val_ids = self.frame_ids[train_count:train_count+val_count]
        self.test_ids = self.frame_ids[train_count+val_count:]


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        elif self.mode == 'val':
            return len(self.val_ids)
        else:
            return len(self.test_ids)
    
    def load_data(self, frame_id):
        img_path, label_path = os.path.join(self.root_dir, 'frames', frame_id), os.path.join(self.root_dir, 'labels', frame_id)
        img = cv2.imread(img_path, 0)
        label = cv2.imread(label_path, 0)
        if label == None: #empty frame or label not present
            label = np.zeros(np.shape(img))

        return img, label
    
    def __getitem__(self, idx):

        if self.mode == 'train':
            frame_id = self.train_ids[idx]
        elif self.mode == 'val':
            frame_id = self.val_ids[idx]
        else:
            frame_id = self.test_ids[idx]

        img, label = self.load_data(frame_id)
        # img_3d, label_3d = self.load_data(os.path.join(self.root_dir, patient_id))

        proccessed_out = {'name': frame_id, 'image': img, 'label' : label}

        if self.transforms:
            proccessed_out = self.transforms(proccessed_out)
        
        return proccessed_out


class classifierDataset(Dataset):
    def __init__(self, root_dir, transforms=None, mode=None, split_ratios = [0.8,0.1,0.1]):
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        # Get all patient IDs
        self.frame_ids = natsorted(os.listdir(os.path.join(self.root_dir, 'frames')))
        self.split_ratios = split_ratios

        random.seed(42)
        random.shuffle(self.frame_ids)

        num_frames = len(self.frame_ids)
        train_count, val_count, test_count = [int(ratio * num_frames) for ratio in split_ratios]
        self.train_ids = self.frame_ids[:train_count]
        self.val_ids = self.frame_ids[train_count:train_count+val_count]
        self.test_ids = self.frame_ids[train_count+val_count:]


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_ids)
        elif self.mode == 'val':
            return len(self.val_ids)
        else:
            return len(self.test_ids)
    
    def load_data(self, frame_id):
        img_path, label_path = os.path.join(self.root_dir, 'frames', frame_id), os.path.join(self.root_dir, 'labels', frame_id)
        img = cv2.imread(img_path, 0)
        label = os.path.exists(label_path) #if a corresponding mask is present, return true 
        if label:
            return img, 1.
        else:
            return img, 0.
    
    def __getitem__(self, idx):

        if self.mode == 'train':
            frame_id = self.train_ids[idx]
        elif self.mode == 'val':
            frame_id = self.val_ids[idx]
        else:
            frame_id = self.test_ids[idx]

        img, label = self.load_data(frame_id)
        label = torch.from_numpy(np.array(label)).long()
        # img_3d, label_3d = self.load_data(os.path.join(self.root_dir, patient_id))

        proccessed_out = {'name': frame_id, 'image': img, 'label' : label}

        if self.transforms:
            proccessed_out = self.transforms(proccessed_out)
        
        return proccessed_out
    
