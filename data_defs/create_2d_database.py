import os
from natsort import natsorted
import shutil

def jpg2png(filename):
    return filename.replace('.jpg', '.png')

root_dir = '/home/nthumbav/Downloads/CVAT_MRI'
dest_dir = '/home/nthumbav/Downloads/CVAT_MRI_2D'

if os.path.exists(os.path.join(dest_dir, 'frames')) == 0:
    os.makedirs(os.path.join(dest_dir, 'frames'))

if os.path.exists(os.path.join(dest_dir, 'labels')) == 0:
    os.makedirs(os.path.join(dest_dir, 'labels'))

for i,patient in enumerate(natsorted(os.listdir(root_dir))):
    for filename in os.listdir(os.path.join(root_dir,patient, 'JPEGImages')):
        shutil.copy(os.path.join(root_dir,patient,'JPEGImages',filename), os.path.join(dest_dir, 'frames', str(i) + '_' + filename))
        try:
            shutil.copy(os.path.join(root_dir,patient,'SegmentationClass',jpg2png(filename)), os.path.join(dest_dir, 'labels', str(i) + '_' + jpg2png(filename)))
        except:
            continue