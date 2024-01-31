import math
import random
import torch
from collections import defaultdict
from config import *
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
# from data_defs.dataset_decathlon import get_train_val_test_Dataloaders
# from data_defs.dataset_mri import get_train_val_test_dataloaders, NIFTISegmentationDataset, NIFTIClassificationDataset
# from data_defs.dataset_mri2d import FrameDataset, classifierDataset
# from data_defs.dataset_ct import NIFTISegmentationDatasetCT
from data_defs.dataset_ct2d import FrameDatasetCT
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model_defs.cammodel import CAMModel
from model_defs.classifier import frameClassifier
from model_defs.classifier_3d import VolumeClassifier
from model_defs.unet2d import UNet2D
from model_defs.unet3d import UNet3D
from transforms import all_transforms
from tqdm import tqdm
import os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import imageio
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn.functional as F
from train_ddp import setup
from transforms import all_transforms
from scipy.spatial.distance import directed_hausdorff
from skimage.filters import threshold_otsu
from monai.networks.nets import AHNet, DynUNet, UNet
from monai.networks.layers import Norm
from utils_infer import cleanup, prepare, prepare_async, calculate_dice, calculate_hd, calculate_iou, visualize_results, calculate_f1, calculate_accuracy
import subprocess

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



def test_models(test_dataloader, model_paths, model, rank, randomize = False):
    seed_everything(42)    
    results = defaultdict(lambda: defaultdict(list))
    random_indices = np.random.choice(len(test_dataloader.dataset), 5, replace=False)

    for model_path in tqdm(model_paths):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        classification_correct = 0

        for i, data in tqdm(enumerate(test_dataloader)):
            if (randomize and i in random_indices) or not randomize:
                image, label, original_label = data['image'].to(rank), data['label'].to(rank), data['original_label']
                output = model(image)
                output_sigmoid = torch.sigmoid(output)
                if MODEL_ARCH == 'CAMModel':
                    pred, gt = torch.argmax(output_sigmoid, 1), torch.argmax(label, 1)
                    cam = model.get_cam(target_class = 1)
                    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=original_label.shape[1:], mode='bilinear', align_corners=False).squeeze()
                    classification_correct = 1 if pred == gt else 0
                    output_sigmoid = torch.unsqueeze(cam, 0)
                dice_score = calculate_dice(output_sigmoid.detach().cpu().numpy(), original_label.cpu().numpy())
                hd_score = calculate_hd(output_sigmoid.detach().cpu().numpy(), original_label.cpu().numpy())
                precision, recall, f1_score = calculate_f1(output_sigmoid.detach().cpu().numpy(), original_label.cpu().numpy())
                accuracy_score = calculate_accuracy(output_sigmoid.detach().cpu().numpy(), original_label.cpu().numpy())

                results[model_path]['images'].append(image.cpu().numpy())
                results[model_path]['labels'].append(original_label.cpu().numpy())
                results[model_path]['predictions'].append(output_sigmoid.detach().cpu().numpy())
                results[model_path]['dice_scores'].append(dice_score)
                results[model_path]['hd_scores'].append(hd_score)
                results[model_path]['precision_scores'].append(precision)
                results[model_path]['recall_scores'].append(recall)
                results[model_path]['f1_scores'].append(f1_score)
                results[model_path]['accuracy_scores'].append(accuracy_score)
                if MODEL_ARCH == 'CAMModel':
                    results[model_path]['classification_scores'].append(classification_correct)
                if np.max(original_label.cpu().numpy()) == 0:
                    results[model_path]['empty_labels'].append(i) #for metric calculation purposes

    return results



def main(rank, world_size):
    setup(rank, world_size)
    test_dataloader = prepare(rank, world_size, batch_size=1, input_dims = INPUT_DIMS, classifier = CLASSIFIER, scan_type = SCAN_TYPE, pin_memory=False, num_workers=0)

    if MODEL_ARCH == 'UNet3D': #irrespective of MR or CT
        model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(rank)
    elif MODEL_ARCH == 'VolumeClassifier':
        model = VolumeClassifier(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(rank)
    elif MODEL_ARCH == 'UNet2D':
        model = UNet2D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(rank)
    elif MODEL_ARCH == 'frameClassifier':
        model = frameClassifier(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(rank)
    elif MODEL_ARCH == 'AHNet':
        model = AHNet(spatial_dims= 2, in_channels=IN_CHANNELS , out_channels= NUM_CLASSES).to(rank)
    elif MODEL_ARCH == 'DynUNet':
        kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        strides = [1, 2, 2, 2, 1]  
        upsample_kernel_size = [2, 2, 2, 2]
        model = DynUNet(spatial_dims= 2, in_channels=IN_CHANNELS , out_channels= NUM_CLASSES, kernel_size=kernel_size, strides=strides, upsample_kernel_size=upsample_kernel_size).to(rank)
    elif MODEL_ARCH == 'MONAIUNet':
        model = UNet(spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(rank)
    else:
        raise NotImplementedError

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # model.load_state_dict(torch.load('checkpoints/ct_2d_segmentation_40_original_labels.pth'))
    # model_paths = ['checkpoints/ct_2d_segmentation_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_10_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_20_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_30_original_labels.pth']
    model_paths = ['checkpoints/ct_2d_segmentation_spline_10_original_labels_512.pth'] #test single model
    results = test_models(test_dataloader, model_paths, model, rank, randomize = False)

    visualize_results(results)
    keys = results[list(results.keys())[0]].keys()
    for key in list(keys):
        if 'scores' in key and ('dice' in key or 'iou' in key or 'hd' in key):
            print('Mean of {} :'.format(key), np.mean([score for i, score in enumerate(results[list(results.keys())[0]][key]) if i not in results[list(results.keys())[0]]['empty_labels']]))
            print('Std of {} :'.format(key), np.std([score for i, score in enumerate(results[list(results.keys())[0]][key]) if i not in results[list(results.keys())[0]]['empty_labels']]))
        
        elif 'scores' in key:
            print('Mean of {} :'.format(key), np.mean(results[list(results.keys())[0]][key]))
            print('Std of {} :'.format(key), np.std(results[list(results.keys())[0]][key]))

    cleanup()


def main_async():
    test_dataloader = prepare_async(batch_size=1, input_dims = INPUT_DIMS, classifier = CLASSIFIER, scan_type = SCAN_TYPE, pin_memory=False, num_workers=0)

    if MODEL_ARCH == 'UNet3D': #irrespective of MR or CT
        model = UNet3D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    elif MODEL_ARCH == 'VolumeClassifier':
        model = VolumeClassifier(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    elif MODEL_ARCH == 'UNet2D':
        model = UNet2D(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    elif MODEL_ARCH == 'frameClassifier':
        model = frameClassifier(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    elif MODEL_ARCH == 'AHNet':
        model = AHNet(spatial_dims= 2, in_channels=IN_CHANNELS , out_channels= NUM_CLASSES).to(device)
    elif MODEL_ARCH == 'DynUNet':
        kernel_size = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
        strides = [1, 2, 2, 2, 1]  
        upsample_kernel_size = [2, 2, 2, 2]
        model = DynUNet(spatial_dims= 2, in_channels=IN_CHANNELS , out_channels= NUM_CLASSES, kernel_size=kernel_size, strides=strides, upsample_kernel_size=upsample_kernel_size).to(device)
    elif MODEL_ARCH == 'MONAIUNet':
        model = UNet(spatial_dims=2, in_channels=1, out_channels=NUM_CLASSES, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
    elif MODEL_ARCH == 'CAMModel':
        model = CAMModel(in_channels=IN_CHANNELS , num_classes= NUM_CLASSES).to(device)
    else:
        raise NotImplementedError

    # model.load_state_dict(torch.load('checkpoints/ct_2d_segmentation_40_original_labels.pth'))
    # model_paths = ['checkpoints/ct_2d_segmentation_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_10_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_20_original_labels.pth',
    # 'checkpoints/ct_2d_segmentation_30_original_labels.pth']
    model_paths = ['checkpoints/ct_2d_segmentation_25_spline_75_512.pth'] #test single model
    results = test_models(test_dataloader, model_paths, model, device, randomize = False)

    # visualize_results(results)
    keys = results[list(results.keys())[0]].keys()
    for key in list(keys):
        if 'scores' in key and ('dice' in key or 'iou' in key or 'hd' in key):
            print('Mean of {} :'.format(key), np.mean([score for i, score in enumerate(results[list(results.keys())[0]][key]) if i not in results[list(results.keys())[0]]['empty_labels']]))
            print('Std of {} :'.format(key), np.std([score for i, score in enumerate(results[list(results.keys())[0]][key]) if i not in results[list(results.keys())[0]]['empty_labels']]))
        elif 'scores' in key:
            print('Mean of {} :'.format(key), np.mean(results[list(results.keys())[0]][key]))
            print('Std of {} :'.format(key), np.std(results[list(results.keys())[0]][key]))



if __name__ == '__main__':
    if RUN_DDP:
        world_size = 1
        mp.spawn(
            main,
            args=(world_size,),
            nprocs=world_size
        )   
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        main_async()