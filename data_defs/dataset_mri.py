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

class NIFTISegmentationDataset(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None, source = 'florida'):
        super(NIFTISegmentationDataset, self).__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.source = source
        if self.source == 'duke':
            self.root_dir = "/home/nthumbav/Downloads/MRI_exps/"
        elif self.source == 'florida':
            self.root_dir = "/home/nthumbav/Downloads/anonymized/PHI_cleared"
        # Get all patient IDs
        if self.source == 'duke':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(os.path.join(root_dir, "imagesTr_Auto3dseg"))]
        elif self.source == 'florida':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(root_dir) if "T2" in filename]

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
        
        img_path = os.path.join(self.root_dir, "imagesTr_Auto3dseg", patient_id + ".nii.gz")
        label_path = os.path.join(self.root_dir, "labelsTr_Auto3dseg", patient_id + ".nii.gz")
        
        img_3d = self.load_nifti_volume(img_path)
        label_3d = self.load_nifti_volume(label_path)

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

class NIFTIInferenceDataset(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None, source = 'florida'):
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.source = source
        if self.source == 'duke':
            self.root_dir = "/home/nthumbav/Downloads/MRI_exps/"
        elif self.source == 'florida':
            self.root_dir = "/home/nthumbav/Downloads/anonymized/PHI_cleared"
        # Get all patient IDs
        if self.source == 'duke':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(os.path.join(self.root_dir, "imagesTr_Auto3dseg"))]
        elif self.source == 'florida':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(self.root_dir) if "T2" in filename]

        self.patient_ids = natsorted(self.patient_ids)

    def __len__(self):
        return len(self.patient_ids)
    
    def load_nifti_volume(self, path):
        nifti_data = nib.load(path)
        volume_3d = np.array(nifti_data.dataobj, dtype=np.float32)
        return volume_3d
    
    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]

        if self.source == 'duke':
            img_path = os.path.join(self.root_dir, "imagesTr_Auto3dseg", patient_id + ".nii.gz")
        elif self.source == 'florida':
            img_path = os.path.join(self.root_dir, patient_id + ".nii.gz")


        img_3d = self.load_nifti_volume(img_path)

        if self.source == 'florida':
            if img_3d.shape[1] > img_3d.shape[2]:
                img_3d = img_3d.transpose((1,0,2))
            img_3d = img_3d.transpose((2,0,1))

        processed_out = {'name': patient_id, 'image': img_3d}

        if self.transforms:
            processed_out = self.transforms(processed_out)
        
        return processed_out


class FrameInferenceDataset(Dataset):
    def __init__(self, root_dir, transforms=None, mode=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        # Get all patient IDs
        self.patient_ids = natsorted(os.listdir(os.path.join(self.root_dir)))

    def __len__(self):
        return len(self.patient_ids)
    
    def load_data(self, path):
        img_frames = os.listdir(os.path.join(path, 'JPEGImages'))
        img_3d, label_3d = [], []

        for filename in os.listdir(os.path.join(path, 'JPEGImages')):
            img_3d.append(cv2.imread(os.path.join(path, 'JPEGImages', filename), cv2.IMREAD_GRAYSCALE))
            if not os.path.exists(os.path.join(path, 'SegmentationClass', filename)):
                label_3d.append(np.zeros(np.shape(img_3d[-1])))
            else:
                label_3d.append(cv2.imread(os.path.join(path, 'SegmentationClass', filename)))
                label_3d[label_3d != 0] = 1 #convert to grayscale labels
        
        return np.array(img_3d), np.array(label_3d)

    
    def __getitem__(self, idx):

        patient_id = self.patient_ids[idx]

        img_3d, label_3d = self.load_data(os.path.join(self.root_dir, patient_id))

        # if self.source == 'florida':
        #     if img_3d.shape[1] > img_3d.shape[2]:
        #         img_3d = img_3d.transpose((1,0,2))
        #     img_3d = img_3d.transpose((2,0,1))

        processed_out = {'name': patient_id, 'image': img_3d, 'label' : label_3d}

        if self.transforms:
            processed_out = self.transforms(processed_out)
        
        return processed_out

class NIFTIClassificationDataset(Dataset):
    def __init__(self, root_dir, split_ratios=[0.8, 0.1, 0.1], transforms=None, mode=None, source = 'florida'):
        super(NIFTIClassificationDataset, self).__init__()
        
        self.root_dir = root_dir
        self.transforms = transforms
        self.mode = mode
        self.source = source
        if self.source == 'duke':
            self.root_dir = "/home/nthumbav/Downloads/MRI_exps/"
        elif self.source == 'florida':
            self.root_dir = "/home/nthumbav/Downloads/anonymized/PHI_cleared"
        # Get all patient IDs
        if self.source == 'duke':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(os.path.join(root_dir, "imagesTr_Auto3dseg"))]
        elif self.source == 'florida':
            self.patient_ids = [filename.replace(".nii.gz", "") for filename in os.listdir(root_dir) if "T2" in filename]

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
        
        img_path = os.path.join(self.root_dir, "imagesTr_Auto3dseg", patient_id + ".nii.gz")
        
        img_3d = self.load_nifti_volume(img_path)

        if 'CAIPI' not in img_path:
            label = 1.
        else:
            label = 0.

        processed_out = {'name': patient_id, 'image': img_3d, 'label': label}

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


def get_train_val_test_dataloaders(root_dir, train_transforms, val_transforms, test_transforms, split_ratios=[0.8, 0.1, 0.1]):
    # Initialize the datasets with appropriate modes
    seed_everything()
    train_set = NIFTISegmentationDataset(root_dir, split_ratios=split_ratios, transforms=[train_transforms, val_transforms, test_transforms], mode='train')
    val_set = NIFTISegmentationDataset(root_dir, split_ratios=split_ratios, transforms=[train_transforms, val_transforms, test_transforms], mode='val')
    test_set = NIFTISegmentationDataset(root_dir, split_ratios=split_ratios, transforms=[train_transforms, val_transforms, test_transforms], mode='test')
    
    # Initialize data loaders
    train_dataloader = DataLoader(train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=VAL_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader