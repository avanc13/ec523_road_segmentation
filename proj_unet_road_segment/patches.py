'''f
rom utils import imgstitch
from PIL import Image
import os
test_fol = '/projectnb/ec523/projects/proj_unet_road_segment/mask'
_, test_folders, _ = next(os.walk(test_fol))
for i in test_folders:
    results_dir = os.path.join(test_fol, str(i))
    imgstitch(results_dir)
'''
import os

folder_path = '/projectnb/ec523/projects/Proj_road_segment_fix/EE8204-ResUNet/dataset/test_patch/1/image/'
num_files = len(os.listdir(folder_path))

print(f"Number of files in folder: {num_files}")