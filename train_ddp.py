'''
Author : Nishanth
'''

import math
import torch
from config import *
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
# from data_defs.dataset_decathlon import get_train_val_test_Dataloaders
# from data_defs.dataset_mri import get_train_val_test_dataloaders, NIFTISegmentationDataset, NIFTIClassificationDataset
# from data_defs.dataset_mri2d import FrameDataset, classifierDataset
# from data_defs.dataset_ct import NIFTISegmentationDatasetCT
from data_defs.dataset_ct2d import FrameDatasetCT, FrameDatasetCTClassification
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from model_defs.cammodel import CAMModel
from model_defs.unet3d import UNet3D
from model_defs.unet2d import UNet2D
from model_defs.classifier_3d import VolumeClassifier
from model_defs.classifier import frameClassifier
from transforms import all_transforms
from losses import DiceLoss, DiscrepancyLoss
from utils_infer import monte_carlo_dropout
from tqdm import tqdm
import os
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from monai.networks.nets import AHNet, DynUNet, UNet
from monai.networks.layers import Norm

torch.autograd.set_detect_anomaly(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def prepare(rank, world_size, batch_size=1, input_dims = 2, classifier = False, scan_type = None, pin_memory=False, num_workers=0):
    if scan_type == 'ct' and input_dims == 3:
        train_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation_multiclass']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
        val_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation_multiclass']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='val')
    
    elif scan_type == 'ct' and input_dims == 2:
        train_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation']['train'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='train')
        val_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation']['train'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='val')

    elif scan_type == 'us' and input_dims == 2:
        train_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation_busi']['train'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='train')
        val_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation_busi']['train'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='val')


    elif input_dims == 3 and not classifier: 
        train_set = NIFTISegmentationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
        val_set = NIFTISegmentationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='val')
    
    elif input_dims == 3 and classifier:
        train_set = NIFTIClassificationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_classification']['train'], all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val']], mode='train')
        val_set = NIFTIClassificationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_classification']['train'], all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val']], mode='val')
    
    elif input_dims == 2 and not classifier: 
        train_set = FrameDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[0.8, 0.2, 0], transforms = all_transforms['2d_segmentation']['train'], mode = 'train')
        val_set = FrameDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[0.8, 0.2, 0], transforms = all_transforms['2d_segmentation']['val'], mode = 'val')

    elif input_dims == 2 and classifier:
        train_set = classifierDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[0.8, 0.2, 0], transforms = all_transforms['2d_classification']['train'], mode = 'train')
        val_set = classifierDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[0.8, 0.2, 0], transforms = all_transforms['2d_classification']['val'], mode = 'val')

    else:
        raise NotImplementedError

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=train_sampler)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=val_sampler)
    
    return train_dataloader, val_dataloader

def prepare_async(batch_size=1, input_dims = 2, classifier = False, scan_type = None, pin_memory=False, num_workers=0):
    if scan_type == 'ct' and input_dims == 3:
        train_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation_multiclass']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
        val_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['3d_segmentation_multiclass']['train'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='val')
    
    elif scan_type == 'ct' and input_dims == 2 and not MODEL_ARCH == 'CAMModel':
        train_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation']['train'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='train')
        val_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation']['train'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='val')

    elif scan_type == 'ct' and input_dims == 2:
        train_set = FrameDatasetCTClassification("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_classification']['train'], all_transforms['2d_classification']['val'], all_transforms['2d_classification']['val']], mode='train')
        val_set = FrameDatasetCTClassification("/home/nthumbav/Downloads/Liver_tumor_CT_2D/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_classification']['train'], all_transforms['2d_classification']['val'], all_transforms['2d_classification']['val']], mode='val')

    elif scan_type == 'us' and input_dims == 2:
        train_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation_busi']['train'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='train')
        val_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/train/", split_ratios=[0.8, 0.2, 0], transforms=[all_transforms['2d_segmentation_busi']['train'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='val')

    else:
        raise NotImplementedError
  
    train_dataloader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)
    
    return train_dataloader, val_dataloader

def cleanup():
    dist.destroy_process_group()

if BACKGROUND_AS_CLASS: NUM_CLASSES += 1

writer = SummaryWriter("runs")

def main(rank, world_size):
    setup(rank, world_size)
    train_dataloader, val_dataloader = prepare(rank, world_size, batch_size=BATCH_SIZE, input_dims = INPUT_DIMS, classifier = CLASSIFIER, scan_type = SCAN_TYPE, pin_memory=False, num_workers=0)
    # train_dataloader, val_dataloader = prepare(rank, world_size, input_dims = INPUT_DIMS, classifier = CLASSIFIER)

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

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    if LOSS_FN == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_LOSS_WEIGHTS).to(rank))
    elif LOSS_FN == 'DiceLoss':
        criterion = DiceLoss()
    elif LOSS_FN == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()
    elif LOSS_FN == 'DiscrepancyLoss':
        criterion_seg = BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_LOSS_WEIGHTS).to(rank))
        criterion_disc = DiscrepancyLoss()

    else:
        raise NotImplementedError
    # criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
    optimizer = Adam(params=model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.3, verbose=True)

    if TRANSFER_WEIGHTS is not None:
        model.load_state_dict(torch.load(TRANSFER_WEIGHTS))


    min_valid_loss = math.inf
    patience_counter = 0

    for epoch in tqdm(range(TRAINING_EPOCH)):

        train_dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
    
        train_loss = 0.0
        model.train()
        for data in tqdm(train_dataloader):
            image, ground_truth, spline_labels = data['image'].to(rank), data['label'].to(rank), data['spline_labels'].to(rank)
            optimizer.zero_grad()
            target = model(image)
            if LOSS_FN == 'CrossEntropyLoss':
                loss = criterion(torch.unsqueeze(target, 0), ground_truth)
            elif LOSS_FN == 'DiscrepancyLoss':
                loss_seg = criterion_seg(target, ground_truth)
                preds = monte_carlo_dropout(model, image, N_MC_SAMPLES)
                annotation_uncertainty = spline_labels.var(dim=1)
                loss_disc = criterion_disc(preds, annotation_uncertainty)
                loss = DISCREPANCY_LOSS_WEIGHTS[0]*loss_seg + DISCREPANCY_LOSS_WEIGHTS[1]*loss_disc
            else:
                loss = criterion(target, ground_truth)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()
        for data in tqdm(val_dataloader):
            image, ground_truth = data['image'].to(rank), data['label'].to(rank)
            
            target = model(image)
            target_sigmoid = torch.sigmoid(target) #for debugging purposes
            if LOSS_FN == 'CrossEntropyLoss':
                loss = criterion(torch.unsqueeze(target, 0), ground_truth)
            elif LOSS_FN == 'DiscrepancyLoss':
                loss_seg = criterion_seg(target, ground_truth)
                preds = monte_carlo_dropout(model, image, N_MC_SAMPLES)
                annotation_uncertainty = data['spline_labels'].var(dim=1)
                loss_disc = criterion_disc(preds, annotation_uncertainty)
                loss = DISCREPANCY_LOSS_WEIGHTS[0]*loss_seg + DISCREPANCY_LOSS_WEIGHTS[1]*loss_disc
            else:
                loss = criterion(target, ground_truth)
            valid_loss += loss.item()
        
        scheduler.step(valid_loss / len(val_dataloader))
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
        
        if min_valid_loss > valid_loss:
            patience_counter = 0
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/{MODEL_SIGNATURE}.pth')
        else:
            patience_counter += 1
            if patience_counter == PATIENCE:
                print('Early stopping activated at epoch {epoch}')
                break

    writer.flush()
    writer.close()
    cleanup()

def main_async():
    '''
    Alternative to main() without using DDP
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, val_dataloader = prepare_async(batch_size=BATCH_SIZE, input_dims = INPUT_DIMS, classifier = CLASSIFIER, scan_type = SCAN_TYPE, pin_memory=False, num_workers=0)
    # train_dataloader, val_dataloader = prepare(rank, world_size, input_dims = INPUT_DIMS, classifier = CLASSIFIER)

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

    if LOSS_FN == 'BCEWithLogitsLoss':
        criterion = BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_LOSS_WEIGHTS).to(device))
    elif LOSS_FN == 'DiceLoss':
        criterion = DiceLoss()
    elif LOSS_FN == 'CrossEntropyLoss':
        criterion = CrossEntropyLoss()
    elif LOSS_FN == 'DiscrepancyLoss':
        criterion_seg = BCEWithLogitsLoss(pos_weight=torch.tensor(BCE_LOSS_WEIGHTS).to(device))
        criterion_disc = DiscrepancyLoss()

    else:
        raise NotImplementedError
    # criterion = CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
    optimizer = Adam(params=model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.3, verbose=True)

    if TRANSFER_WEIGHTS is not None:
        model.load_state_dict(torch.load(TRANSFER_WEIGHTS))


    min_valid_loss = math.inf
    patience_counter = 0

    for epoch in tqdm(range(TRAINING_EPOCH)):

    
        train_loss = 0.0
        model.train()
        for data in tqdm(train_dataloader):
            image, ground_truth = data['image'].to(device), data['label'].to(device)
            optimizer.zero_grad()
            target = model(image)
            if LOSS_FN == 'CrossEntropyLoss':
                loss = criterion(torch.unsqueeze(target, 0), ground_truth)
            elif LOSS_FN == 'DiscrepancyLoss':
                spline_labels = data['spline_labels'].to(device)
                loss_seg = criterion_seg(target, ground_truth)
                preds = monte_carlo_dropout(model, image, N_MC_SAMPLES)
                annotation_uncertainty = spline_labels.var(dim=1)
                loss_disc = criterion_disc(preds, annotation_uncertainty)
                loss = DISCREPANCY_LOSS_WEIGHTS[0]*loss_seg + DISCREPANCY_LOSS_WEIGHTS[1]*loss_disc
            else:
                loss = criterion(target, ground_truth)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        valid_loss = 0.0
        model.eval()
        for data in tqdm(val_dataloader):
            image, ground_truth = data['image'].to(device), data['label'].to(device)
            
            target = model(image)
            target_sigmoid = torch.sigmoid(target) #for debugging purposes
            if LOSS_FN == 'CrossEntropyLoss':
                loss = criterion(torch.unsqueeze(target, 0), ground_truth)
            elif LOSS_FN == 'DiscrepancyLoss':
                loss_seg = criterion_seg(target, ground_truth)
                preds = monte_carlo_dropout(model, image, N_MC_SAMPLES)
                annotation_uncertainty = data['spline_labels'].var(dim=1)
                loss_disc = criterion_disc(preds, annotation_uncertainty)
                loss = DISCREPANCY_LOSS_WEIGHTS[0]*loss_seg + DISCREPANCY_LOSS_WEIGHTS[1]*loss_disc
            else:
                loss = criterion(target, ground_truth)
            valid_loss += loss.item()
        
        scheduler.step(valid_loss / len(val_dataloader))
        writer.add_scalar("Loss/Train", train_loss / len(train_dataloader), epoch)
        writer.add_scalar("Loss/Validation", valid_loss / len(val_dataloader), epoch)
        
        print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(train_dataloader)} \t\t Validation Loss: {valid_loss / len(val_dataloader)}')
        
        if min_valid_loss > valid_loss:
            patience_counter = 0
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(model.state_dict(), f'checkpoints/{MODEL_SIGNATURE}.pth')
        else:
            patience_counter += 1
            if patience_counter == PATIENCE:
                print('Early stopping activated at epoch {epoch}')
                break


if __name__ == '__main__':
    if not RUN_DDP:
        main_async()
    else:
        world_size = 1
        mp.spawn(
            main,
            args=(world_size,),
            nprocs=world_size
        )