# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 02:44:43 2020

@author: edwin.p.alegre
"""
from PIL import Image
import os
import numpy as np
from skimage.io import imread
from tqdm import tqdm

OGIMG_SIZE = 1500
IMG_SIZE = 224
OVERLAP = 14

def imgstitch(img_path):
    """
    This function overlays the predicted image patches with each other to produce the final output image.
    It should be noted that the overlap regions are INTENDED to be averaged to minimize error. This is not
    done in this function. They are just overwritten by the next image patch. This will be implemented in the future 
    for a more robust approach
    
    Parameters
    ----------
    img_path : STRING
        Path to directory with test image patches

    Returns
    -------
    None. The final stitched image will be automatically saved in the same directory as the image patches with the 
    name 'ouptut.png''

    """
    _, _, img_files = next(os.walk(img_path))
    
    img_files = sorted(img_files,key=lambda x: int(os.path.splitext(x)[0]))
    IMG_WIDTH, IMG_HEIGHT = (Image.open(img_path + '/11.png')).size
    
    img = np.zeros((len(img_files), IMG_WIDTH, IMG_HEIGHT), dtype=np.uint8)
    full_img = Image.new('RGB', (1470, 1470))
    x, y = (0, 0)
    
    for n, id_ in enumerate(img_files):
        img = Image.open(img_path + '/' + str(id_))
        if x < 1460:
            full_img.paste(img, (x, y))
            x += IMG_WIDTH - OVERLAP
        if x > 1460:
            x = 0
            y += IMG_WIDTH - OVERLAP
            full_img.paste(img, (x, y))
    
    full_img.save(os.path.join(img_path, 'output') + '.png', 'PNG')
    
def DatasetLoad(train_dataset, test_dataset, val_dataset, max_train_samples=100):
    """
    Loads training and test datasets. Validation is skipped (returns None).
    """
    IMG_SIZE = 224  # Set explicitly in case not global

    ### TRAINING DATASET ###
    _, _, train_files = next(os.walk(os.path.join(train_dataset, 'image')))
    training_imgs = len(train_files)
    train_ids = list(range(1, min(training_imgs + 1, max_train_samples + 1)))
    

    X_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids), desc="Loading training patches"):
        img = imread(os.path.join(train_dataset, 'image', f'{id_}.png')).astype(np.float32) / 255.0
        mask = imread(os.path.join(train_dataset, 'mask', f'{id_}.png')).astype(np.float32)
        #mask = (mask > 127).astype(np.float32)  # This is the missing line
        mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
        # Ensure values are clipped to [0, 255] before binarizing
        mask = np.clip(mask, 0, 255)
        mask = (mask>127).astype(np.float32)
        X_train[n] = img
        Y_train[n] = np.expand_dims(mask, axis=-1).astype(np.float32)
    assert np.isfinite(X_train).all(), "NaNs or Infs in X_train"
    assert np.isfinite(Y_train).all(), "NaNs or Infs in Y_train"
    assert set(np.unique(Y_train)) <= {0.0, 1.0}, f"Unexpected Y_train values: {np.unique(Y_train)}"

    ### TESTING DATASET ###
    ### TESTING DATASET ###
    _, test_fol, _ = next(os.walk(test_dataset))
    X_test = {}
    Y_test = {}

    for folder in test_fol:
        test_path = os.path.join(test_dataset, folder)
        
        image_dir = os.path.join(test_path, "image")
        mask_dir = os.path.join(test_path, "mask")
        
        test_files_sorted = sorted(
            os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0])
        )
        mask_files_sorted = sorted(
            os.listdir(mask_dir), key=lambda x: int(os.path.splitext(x)[0])
        )
        
        num_test_samples = min(len(test_files_sorted), len(mask_files_sorted))
        X_test[folder] = np.zeros((num_test_samples, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        Y_test[folder] = np.zeros((num_test_samples, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
        
        for n in tqdm(range(num_test_samples), desc=f"Loading test patches: {folder}"):
            img = imread(os.path.join(image_dir, test_files_sorted[n])).astype(np.float32) / 255.0
            mask = imread(os.path.join(mask_dir, mask_files_sorted[n])).astype(np.float32)
            mask = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
            # Ensure values are clipped to [0, 255] before binarizing
            mask = np.clip(mask, 0, 255)
            mask = (mask > 127).astype(np.float32)  # Binarize
            X_test[folder][n] = img
            Y_test[folder][n] = np.expand_dims(mask, axis=-1).astype(np.float32)
    for folder in X_test:
        assert np.isfinite(X_test[folder]).all(), f"NaNs or Infs in X_test for {folder}"
        assert np.isfinite(Y_test[folder]).all(), f"NaNs or Infs in Y_test for {folder}"
    for folder in Y_test:
        u = np.unique(Y_test[folder])
        assert set(u) <= {0.0, 1.0}, f"Unexpected values in Y_test[{folder}]: {u}"



   
    return X_train, Y_train, X_test, Y_test, None, None
    