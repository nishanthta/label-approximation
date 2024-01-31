import os
import random
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import binary_erosion, measurements
from scipy.spatial.distance import pdist
from matplotlib.patches import Circle
from skimage.draw import disk
import scipy.stats as stats
import torch
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

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


def create_gaussian_label_volume(label_volume):
    """
    Create a label volume where each slice is a mask depicting Gaussian distributions centered at tumor locations.

    Parameters:
    label_volume (numpy.ndarray): The 3D binary mask volume for tumor segmentation.

    Returns:
    numpy.ndarray: The 3D binary mask volume with Gaussian approximations of tumors.
    """
    # Initialize the Gaussian label volume with zeros
    gaussian_label_volume = np.zeros_like(label_volume, dtype=np.float32)

    if len(label_volume.shape) == 2:
        label_slice = label_volume
        tumor_features = extract_tumor_features(label_slice)

        for (center, diameter) in tumor_features:
            x = np.arange(0, label_slice.shape[1], 1, float)
            y = np.arange(0, label_slice.shape[0], 1, float)
            y = y[:, np.newaxis]

            x0, y0 = center  # Tumor center
            sigma = diameter / 2  # Standard deviation

            # Create 2D Gaussian
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

            # Normalize to make the peak equal 1
            g /= np.max(g)

            # Add the Gaussian to the current slice
            gaussian_label_volume[:, :] += g

    else:
        # Iterate over each slice of the label volume
        for slice_index in range(label_volume.shape[0]):
            label_slice = label_volume[slice_index, :, :]
            tumor_features = extract_tumor_features(label_slice)

            # Generate a Gaussian distribution for each tumor
            for (center, diameter) in tumor_features:
                x = np.arange(0, label_slice.shape[1], 1, float)
                y = np.arange(0, label_slice.shape[0], 1, float)
                y = y[:, np.newaxis]

                x0, y0 = center  # Tumor center
                sigma = diameter / 2  # Standard deviation

                # Create 2D Gaussian
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

                # Normalize to make the peak equal 1
                g /= np.max(g)

                # Add the Gaussian to the current slice
                gaussian_label_volume[slice_index, :, :] += g

    gaussian_label_volume = np.clip(gaussian_label_volume, 0, 1)

    return gaussian_label_volume

def create_circular_label_volume(label_volume, approximation_error=None):
    """
    Create an approximated circular label volume where each slice is a binary mask depicting the circles denoted by the computed center and radii.

    Parameters:
    label_volume (numpy.ndarray): The 3D binary mask volume for tumor segmentation.

    Returns:
    numpy.ndarray: The 3D binary mask volume with approximated tumors as circles.
    """
    if approximation_error is not None:
        center_error_range, diameter_error_percentage = approximation_error[0], approximation_error[1]
    else:
        center_error_range, diameter_error_percentage = None, None
    approximated_label_volume = np.zeros_like(label_volume)

    def apply_center_error(center, error_range):
        return tuple(c + random.randint(-error_range, error_range) for c in center)

    # Function to apply diameter error
    def apply_diameter_error(diameter, error_percentage):
        # error_factor = 1 + error_percentage * (2 * random.random() - 1)
        error_factor = diameter*error_percentage*0.01
        if random.random() > 0.5:
            return diameter + error_factor
        else:
            return diameter - error_factor

    if len(label_volume.shape) == 2:  # Single slice has been passed
        label_slice = label_volume
        tumor_features = extract_tumor_features(label_slice)
        for (center, diameter) in tumor_features:
            # Apply center error if specified
            if center_error_range is not None:
                center = apply_center_error(center, center_error_range)
            
            # Apply diameter error if specified
            if diameter_error_percentage is not None:
                diameter = apply_diameter_error(diameter, diameter_error_percentage)

            radius = diameter / 2
            rr, cc = disk((int(center[0]), int(center[1])), int(radius), shape=label_slice.shape)
            approximated_label_volume[rr, cc] = 1


    else:
        for slice_index in range(label_volume.shape[0]):
            label_slice = label_volume[slice_index, :, :]
            tumor_features = extract_tumor_features(label_slice)

            # Draw each tumor as a circle in the approximated label volume
            for (center, diameter) in tumor_features:
                radius = diameter / 2
                rr, cc = disk((center[0], center[1]), radius, shape=label_slice.shape)
                approximated_label_volume[slice_index, rr, cc] = 1

    return approximated_label_volume

def extract_tumor_features(label_volume):
    """
    Extract the center location and diameter of tumors in the slice with the largest tumor area.

    Parameters:
    label_volume (numpy.ndarray): The 3D binary mask volume for tumor segmentation.

    Returns:
    list of tuples: Each tuple contains the center (row, col) and the diameter of a tumor.
    """

    if len(label_volume.shape) == 2: #input is a tumor slice
        labeled_tumor_regions, num_features = measurements.label(label_volume)
        centers_of_mass = measurements.center_of_mass(label_volume, labeled_tumor_regions, range(1, num_features+1))
        tumor_features = []
        for i in range(1, num_features + 1):
            # Find the coordinates of the tumor region
            region_coords = np.argwhere(labeled_tumor_regions == i)
            # Calculate the diameter using the pairwise distance between points
            if len(region_coords) == 1:
                continue
            region_diameter = np.max(pdist(region_coords))
            tumor_features.append((centers_of_mass[i - 1], region_diameter))
 

    else:
        tumor_areas = np.sum(label_volume, axis=(1, 2))
        # Find the index of the slice with the largest tumor area
        largest_tumor_slice_index = np.argmax(tumor_areas)

        largest_label_slice = label_volume[largest_tumor_slice_index, :, :]

        # Label each connected tumor region
        labeled_tumor_regions, num_features = measurements.label(largest_label_slice)

        # Calculate the center of mass for each tumor region
        centers_of_mass = measurements.center_of_mass(largest_label_slice, labeled_tumor_regions, range(1, num_features+1))

        # Calculate the diameter of each tumor region
        tumor_features = []
        for i in range(1, num_features + 1):
            # Find the coordinates of the tumor region
            region_coords = np.argwhere(labeled_tumor_regions == i)
            # Calculate the diameter using the pairwise distance between points
            region_diameter = np.max(pdist(region_coords))
            tumor_features.append((centers_of_mass[i - 1], region_diameter))
 
    return tumor_features


def calculate_approximation_error(input_volume, label_volume, savepath):
    """
    Compute the center and diameter of the tumors, approximate the tumor with a circle,
    compare that circle with the original tumor outline, and display the slice with the maximum approximation error.

    Parameters:
    input_volume (numpy.ndarray): The 3D CT input volume as a NumPy array.
    label_volume (numpy.ndarray): The 3D binary mask volume for tumor segmentation.
    """
    # Calculate the area of the tumor on each slice
    tumor_areas = np.sum(label_volume, axis=(1, 2))
    largest_tumor_slice_index = np.argmax(tumor_areas)
    largest_label_slice = label_volume[largest_tumor_slice_index, :, :]
    largest_input_slice = input_volume[largest_tumor_slice_index, :, :]

    # Get the tumor features for the largest slice
    tumor_features = extract_tumor_features(label_volume)

    # Calculate the error between the actual tumor area and the approximated circle area
    errors = []
    for center, diameter in tumor_features:
        radius = diameter / 2
        approximated_area = np.pi * (radius ** 2)
        actual_area = np.sum(largest_label_slice[measurements.label(largest_label_slice) == 1])
        error = np.abs(approximated_area - actual_area)
        errors.append(error)

    # Find the tumor with the maximum approximation error
    max_error_index = np.argmax(errors)
    max_error_center, max_error_diameter = tumor_features[max_error_index]
    max_error_radius = max_error_diameter / 2

    # Create the circle approximation for the tumor with the maximum error
    tumor_outline = binary_erosion(largest_label_slice) ^ largest_label_slice
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(largest_input_slice, cmap='gray')
    ax.contour(tumor_outline, colors='r')

    # Add the approximated circle to the plot
    circle = Circle(max_error_center[::-1], max_error_radius, color='blue', fill=False, linestyle='--')
    ax.add_patch(circle)

    # Display the result
    ax.set_title('Max Approximation Error: Tumor Overlay with Circle Approximation')
    ax.axis('off')
    plt.savefig(savepath, dpi=300)

def display_largest_tumor_slice_with_overlay(input_volume, label_volume, savepath):
    """
    Display the input CT slice with an overlay of the tumor mask for the slice with the largest tumor area.

    Parameters:
    input_volume (numpy.ndarray): The 3D CT input volume as a NumPy array.
    label_volume (numpy.ndarray): The 3D binary mask volume for tumor segmentation.
    savepath (str): The path to save the output image.
    """

    # Calculate the area of the tumor on each slice
    tumor_areas = np.sum(label_volume, axis=(1, 2))

    # Find the index of the slice with the largest tumor area
    largest_tumor_slice_index = np.argmax(tumor_areas)

    # Extract the corresponding slices for input and label volumes
    largest_input_slice = input_volume[largest_tumor_slice_index, :, :]
    largest_label_slice = label_volume[largest_tumor_slice_index, :, :]

    # Create an outline for the tumor mask
    eroded_mask = binary_erosion(largest_label_slice)
    outline = largest_label_slice ^ eroded_mask

    # Plot the input slice with an overlay of the tumor outline
    plt.figure(figsize=(6, 6))
    plt.imshow(largest_input_slice, cmap='gray')
    plt.contour(outline, colors='red', linewidths=0.5)  # Draw the outline in red
    plt.title('Tumor Overlay on Input Slice')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(savepath, dpi=300)



def window_image(ct_image, window_center = 70, window_width = 150):
    lower_bound = window_center - window_width / 2
    upper_bound = window_center + window_width / 2

    windowed_image = np.clip(ct_image, lower_bound, upper_bound)

    return (windowed_image - lower_bound) / (upper_bound - lower_bound)


def load_nifti_volume(path):
        # Load NIFTI file and return as a 3D volume
        nifti_data = nib.load(path)
        volume_3d = nifti_data.get_fdata()
      #   volume_3d = np.array(nifti_data.dataobj, dtype=np.float32)
        volume_3d = np.transpose(volume_3d, (2,0,1))
      #   volume_3d = np.rot90(volume_3d)
        return volume_3d

def create_approximated_spline_volume(input_mask, num_points=10, visualize=False):
    def extract_boundary_points(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [cnt.reshape(-1, 2) for cnt in contours]

    def select_points(points, num_points=num_points):
        total_points = len(points)
        selected_indices = np.round(np.linspace(0, total_points - 1, num_points)).astype(int)
        # selected_indices = random.sample(range(total_points), num_points)
        return points[selected_indices,:]

    def fit_spline_to_points(points):
        points = np.array(points, dtype=np.float32).T
        tck, u = splprep([points[0], points[1]], s=0)
        u_new = np.linspace(u.min(), u.max(), 1000)
        x_new, y_new = splev(u_new, tck, der=0)
        return np.array(list(zip(x_new, y_new))).astype(np.int32)

    # Check if the input is a tensor and convert it to a numpy array if true
    input_is_tensor = torch.is_tensor(input_mask)
    if input_is_tensor:
        input_mask = input_mask.detach().cpu().numpy()

    all_boundary_points = extract_boundary_points(input_mask)
    new_mask = np.zeros_like(input_mask)

    if visualize:
        plt.figure(figsize=(8, 8))
        plt.imshow(input_mask, cmap='gray')

    for boundary_points in all_boundary_points:
        selected_points = select_points(boundary_points)
        try:
            spline_points = fit_spline_to_points(selected_points)
            cv2.fillPoly(new_mask, [spline_points], color=(255))
        except:
            pass
        if visualize:
            plt.scatter(selected_points[:, 0], selected_points[:, 1], color='red')
            plt.plot(spline_points[:, 0], spline_points[:, 1], color='blue')

    if visualize:
        plt.title('Tumor Segmentation with Spline Fitted Polygons')
        plt.show()  # Changed from savefig to show for inline display

    # Convert the new_mask back to a tensor if the input was a tensor
    if input_is_tensor:
        new_mask = torch.tensor(new_mask, dtype=torch.uint8)

    return new_mask


def visualize_spline(input_masks):
    fig, axs = plt.subplots(len(input_masks), 4, figsize=(15, 3 * len(input_masks)))

    for i, mask in enumerate(input_masks):
        if np.max(mask) > 0:  # Check if the mask is non-empty
            axs[i, 0].imshow(mask, cmap='gray')
            axs[i, 0].set_title('Original Mask')
            axs[i, 0].axis('off')

            for j, points in enumerate([4, 5, 10], 1):
                approx_mask = create_approximated_spline_volume(mask, num_points=points)
                axs[i, j].imshow(approx_mask, cmap='gray')
                axs[i, j].set_title(f'Approx. {points} Points')
                axs[i, j].axis('off')
        else:
            for ax in axs[i]:
                ax.axis('off')  # Hide axes for empty rows

    plt.tight_layout()
    plt.savefig('scratch/spline_approximations_busi.png', dpi=300)

def load_random_masks(input_dir, num_masks=5):
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.png')]
    
    # Filter out non-empty masks first
    non_empty_files = []
    for file_path in all_files:
        mask = cv2.imread(file_path, 0)
        if np.max(mask) > 0:  # Check if the mask is non-empty
            non_empty_files.append(file_path)

    selected_files = random.sample(non_empty_files, min(num_masks, len(non_empty_files)))

    selected_masks = [cv2.imread(file_path, 0) for file_path in selected_files]

    return selected_masks

if __name__ == '__main__':
    seed_everything(1)

    root_dir = '/home/nthumbav/Downloads/BUSI_2D/Dataset_BUSI_with_GT/test/masks'

    selected_masks = load_random_masks(root_dir)
    visualize_spline(selected_masks)

    pass