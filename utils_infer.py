import torch
from config import *
from data_defs.dataset_ct2d import FrameDatasetCT, FrameDatasetCTClassification
from transforms import all_transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transforms import all_transforms
from scipy.spatial.distance import directed_hausdorff
from skimage.filters import threshold_otsu
import numpy as np

def prepare(rank, world_size, batch_size=1, input_dims = 2, classifier = False, scan_type = None, pin_memory=False, num_workers=0):
    if scan_type == 'ct' and input_dims == 3:
        test_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[1.,0, 0], transforms=[all_transforms['3d_segmentation_multiclass']['val'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
    
    elif scan_type == 'ct' and input_dims == 2:
        test_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/test/", split_ratios=[1.,0, 0], transforms=[all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='train')
    
    elif scan_type == 'us' and input_dims == 2:
        test_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/test/", split_ratios=[1.,0, 0], transforms=[all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='train')
    
    elif input_dims == 3 and not classifier: 
        test_set = NIFTISegmentationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[1.,0, 0], transforms=[all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
    
    elif input_dims == 3 and classifier:
        test_set = NIFTIClassificationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[1.,0, 0], transforms=[all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val']], mode='train')
    
    elif input_dims == 2 and not classifier: 
        test_set = FrameDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[1.,0, 0], transforms = all_transforms['2d_segmentation']['val'], mode = 'train')

    elif input_dims == 2 and classifier:
        test_set = classifierDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[1.,0, 0], transforms = all_transforms['2d_classification']['val'], mode = 'train')

    else:
        raise NotImplementedError

    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    
    test_dataloader = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False, sampler=test_sampler)
    
    return test_dataloader

def prepare_async(batch_size=1, input_dims = 2, classifier = False, scan_type = None, pin_memory=False, num_workers=0):
    if scan_type == 'ct' and input_dims == 3:
        test_set = NIFTISegmentationDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT/", split_ratios=[1.,0, 0], transforms=[all_transforms['3d_segmentation_multiclass']['val'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
    
    elif scan_type == 'ct' and input_dims == 2 and not MODEL_ARCH == 'CAMModel':
        test_set = FrameDatasetCT("/home/nthumbav/Downloads/Liver_tumor_CT_2D/test/", split_ratios=[1.,0, 0], transforms=[all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val'], all_transforms['2d_segmentation']['val']], mode='train')
    
    elif scan_type == 'ct' and input_dims == 2:
        test_set = FrameDatasetCTClassification("/home/nthumbav/Downloads/Liver_tumor_CT_2D/test/", split_ratios=[1.,0, 0], transforms=[all_transforms['2d_classification']['val'], all_transforms['2d_classification']['val'], all_transforms['2d_classification']['val']], mode='train')
    
    elif scan_type == 'us' and input_dims == 2:
        test_set = FrameDatasetCT("/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/test/", split_ratios=[1.,0, 0], transforms=[all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val'], all_transforms['2d_segmentation_busi']['val']], mode='train')
    
    elif input_dims == 3 and not classifier: 
        test_set = NIFTISegmentationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[1.,0, 0], transforms=[all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val'], all_transforms['3d_segmentation']['val']], mode='train')
    
    elif input_dims == 3 and classifier:
        test_set = NIFTIClassificationDataset('/home/nthumbav/Downloads/MRI_exps/', split_ratios=[1.,0, 0], transforms=[all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val'], all_transforms['3d_classification']['val']], mode='train')
    
    elif input_dims == 2 and not classifier: 
        test_set = FrameDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[1.,0, 0], transforms = all_transforms['2d_segmentation']['val'], mode = 'train')

    elif input_dims == 2 and classifier:
        test_set = classifierDataset('/home/nthumbav/Downloads/CVAT_MRI_2D/', split_ratios=[1.,0, 0], transforms = all_transforms['2d_classification']['val'], mode = 'train')

    else:
        raise NotImplementedError

    test_dataloader = DataLoader(test_set, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=False)
    
    return test_dataloader

def cleanup():
    '''
    Clean up GPU cache
    '''
    dist.destroy_process_group()

def calculate_iou(output, original_label):
    # Convert the output to a binary mask
    output_flat = output.ravel()
    threshold = threshold_otsu(output_flat)
    output_mask = output > threshold
    
    # Ensure the original label is a binary mask
    original_mask = original_label > 0.5
    if original_mask.max() == 0:
            return None
    # Calculate Intersection and Union
    intersection = np.logical_and(output_mask, original_mask)
    union = np.logical_or(output_mask, original_mask)

    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def calculate_dice(output, original_label, threshold=0.75):
    # Convert the output to a binary mask based on the threshold
    # threshold = threshold_otsu(output)
    output_mask = output > threshold
    

    # Ensure the original label is a binary mask
    original_mask = original_label > 0.5
    if original_mask.max() == 0:
        return None
    # Calculate the intersection and the sum of both masks
    intersection = np.logical_and(output_mask, original_mask)
    total = np.sum(output_mask) + np.sum(original_mask)

    # Calculate the Dice coefficient
    dice = 2 * np.sum(intersection) / total if total != 0 else 1

    return dice

def calculate_accuracy(output, original_label, threshold=0.75):
    # Convert the output to a binary mask based on the threshold
    # threshold = threshold_otsu(output)
    output_mask = output > threshold
    

    # Ensure the original label is a binary mask
    original_mask = original_label > 0.5
    if original_mask.max() == 0:
        return None
    
    acc = np.sum(output_mask == original_mask)*1. / (np.sum(output_mask == original_mask) + np.sum(output_mask != original_mask)) 

    return acc

def calculate_hd(output, original_label):
    # Convert the output to a binary mask
    threshold = threshold_otsu(output)
    output_mask = output > threshold
    output_mask = output_mask.astype(np.bool)
    

    # Ensure the original label is a binary mask
    original_mask = original_label > 0.5
    original_mask = original_mask.astype(np.bool)
    if original_mask.max() == 0:
            return None
    # Compute Hausdorff Distance
    hd = max(directed_hausdorff(np.argwhere(output_mask), np.argwhere(original_mask))[0], directed_hausdorff(np.argwhere(original_mask), np.argwhere(output_mask))[0])

    return hd

def calculate_f1(output, original_label):
    output_flat = output.ravel()
    threshold = threshold_otsu(output_flat)
    output_mask = output > threshold
    
    original_mask = original_label > 0.5
    if original_mask.max() == 0:
        return None
    
    TP = np.logical_and(output_mask, original_mask).sum()
    FP = np.logical_and(output_mask, np.logical_not(original_mask)).sum()
    FN = np.logical_and(np.logical_not(output_mask), original_mask).sum()
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    if precision + recall == 0:  # Avoid division by zero
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def display_top_5(top_5):
    fig, axs = plt.subplots(5, 4, figsize=(20, 20))

    for i, (iou_score, data) in enumerate(top_5):
        image, label, original_label, output = data

        axs[i, 0].imshow(image[0, 0], cmap='gray')  # Assuming image is in [C, H, W] format
        axs[i, 0].set_title("Input Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(label[0, 0], cmap='gray')  # Assuming label is in [C, H, W] format
        axs[i, 1].set_title("Circular Label")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(original_label[0, 0], cmap='gray')  # Assuming original_label is in [C, H, W] format
        axs[i, 2].set_title(f"Original Label\nIoU: {iou_score:.2f}")
        axs[i, 2].axis('off')

        axs[i, 3].imshow(output[0, 0], cmap='gray')  # Assuming output is in [C, H, W] format
        axs[i, 3].set_title("Model Output")
        axs[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig("scratch/circular_iou.png", dpi = 300)



def monte_carlo_dropout(model, volume, n_mc_samples=5):
    '''
    Run MC Dropout and return the tensor of all output runs
    '''
    model.train()  # need to use dropout
    def set_bn_eval(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)

    targets_list = []  # Use a list to collect tensors

    for i in range(n_mc_samples):
        target = model(volume)
        targets_list.append(torch.sigmoid(target))  # Append to list

    # Concatenate all tensors in the list along a new dimension
    targets = torch.stack(targets_list, dim=1)

    return torch.squeeze(targets, 2)

def visualize_results(results):
    fig, axes = plt.subplots(5, len(list(results.keys())) + 2, figsize=(15, 10))
    model_column_titles = ['Original', 'Trained on 10%', 'Trained on 20%', 'Trained on 30%']

    for i in range(5):
        # Display input image
        axes[i, 0].imshow(results[list(results.keys())[0]]['images'][i][0][0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')

        # Display original label
        axes[i, 1].imshow(results[list(results.keys())[0]]['labels'][i][0][0], cmap='gray')
        axes[i, 1].set_title('Original Label')
        axes[i, 1].axis('off')

        # Loop through each model's results
        for j, model_path in enumerate(results.keys()):
            output = results[model_path]['predictions'][i][0][0]
            threshold = threshold_otsu(output)
            output_mask = output > threshold
            axes[i, j+2].imshow(output_mask, cmap='gray')
            axes[i, j+2].set_title(f"{model_column_titles[j]}\nDice: {results[model_path]['dice_scores'][i]:.2f}")
            axes[i, j+2].axis('off')

    plt.tight_layout()
    plt.savefig('scratch/subset_results.png', dpi=300)
