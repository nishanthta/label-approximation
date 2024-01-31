import os
import random
import shutil

# Set the paths to your data folders
data_folder = '/home/nthumbav/Downloads/Liver_tumor_CT_2D'
train_folder = 'train'
test_folder = 'test'

# Create train and test folders
os.makedirs(os.path.join(data_folder, train_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(data_folder, train_folder, 'masks'), exist_ok=True)
os.makedirs(os.path.join(data_folder, test_folder, 'images'), exist_ok=True)
os.makedirs(os.path.join(data_folder, test_folder, 'masks'), exist_ok=True)

# Create a dictionary to store patient IDs and their corresponding slices
patient_slices = {}

# Iterate through the images folder and collect patient ID and slice information
for filename in os.listdir(os.path.join(data_folder, 'images')):
    patient_id, slice_num = filename.split('_')
    slice_num = slice_num.split('.')[0]  # Remove the file extension
    if patient_id not in patient_slices:
        patient_slices[patient_id] = []
    patient_slices[patient_id].append((filename, f'{patient_id}_{slice_num}.png'))

# Shuffle the patient IDs to randomize the split
patient_ids = list(patient_slices.keys())
random.shuffle(patient_ids)

# Calculate the split point
split_point = int(0.85 * len(patient_ids))

# Assign patients to train and test sets
train_patients = patient_ids[:split_point]
test_patients = patient_ids[split_point:]

# Move images and masks to train and test folders based on patient ID
for patient_id in train_patients:
    for img_filename, mask_filename in patient_slices[patient_id]:
        shutil.move(os.path.join(data_folder, 'images', img_filename),
                    os.path.join(data_folder, train_folder, 'images', img_filename))
        shutil.move(os.path.join(data_folder, 'masks', mask_filename),
                    os.path.join(data_folder, train_folder, 'masks', mask_filename))

for patient_id in test_patients:
    for img_filename, mask_filename in patient_slices[patient_id]:
        shutil.move(os.path.join(data_folder, 'images', img_filename),
                    os.path.join(data_folder, test_folder, 'images', img_filename))
        shutil.move(os.path.join(data_folder, 'masks', mask_filename),
                    os.path.join(data_folder, test_folder, 'masks', mask_filename))

print("Data split into train and test sets successfully.")
