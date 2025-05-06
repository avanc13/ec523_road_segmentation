# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:42:46 2020

@author: edwin.p.alegre
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import model_unet_5
import numpy as np
import tensorflow as tf
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
import random
import scipy.misc
from PIL import Image
import shutil
from utils import DatasetLoad,  relaxed_metrics
from utils_fix_jn import imgstitch

########################### LEARNING RATE SCHEDULER ###########################

# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
def schedlr(epoch, lr):
    new_lr = 0.001 * (0.1)**(floor(epoch/10))
    return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 224
BATCH = [2,4,8,16,32]
EPOCHS = 100

################################### DATASET ###################################

# Paths for relevant datasets to load in
train_dataset = r'/projectnb/ec523/projects/Proj_road_segment_fix/EE8204-ResUNet/dataset/samples_train'
test_dataset = r'/projectnb/ec523/projects/Proj_road_segment_fix/EE8204-ResUNet/dataset/test_patch'
#val_dataset = r'dataset/samples_val'

# Make a list of the test folders to be used when predicting the model. This will be fed into the prediction
# flow to generate the stitched image based off the predictions of the patches fed into the network
_, test_fol, _ = next(os.walk(test_dataset))

# Load in the relevant datasets 
if os.path.exists("preprocessed_data.npz"):
    data = np.load("preprocessed_data.npz", allow_pickle = True)
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test  = data["X_test"]
    Y_test  = data["Y_test"]
    X_val   = data["X_val"]
    Y_val   = data["Y_val"]
else:
    X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset)
    np.savez_compressed("preprocessed_data.npz",
                        X_train=X_train, Y_train=Y_train,
                        X_test=X_test, Y_test=Y_test,
                        X_val=X_val, Y_val=Y_val)  
################################ RESIDUAL UNET ################################
smooth = 1e-6

def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    numerator = 2.0 * intersection + smooth
    denominator = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1) + smooth
    return tf.reduce_mean(numerator / denominator)

def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
sgd_optimizer = Adam()

# Metrics to be used when evaluating the network
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
f1 = F1Score(num_classes=2, name='f1', average='micro', threshold=0.4)

# Instantiate the network 
'''
model = model_unet_5.UNet((IMG_SIZE, IMG_SIZE, 3))
model.compile(optimizer=sgd_optimizer, loss=dice_coef_loss, metrics=['accuracy', precision, recall, f1])

# Callacks to be used in the network. Checkpoint can be adjusted to save the best (lowest loss) if desired. 
checkpoint_path = os.path.join(dname, 'models_5_dice_loss', 'unet.{epoch:02d}-{f1:.2f}.hdf5')
checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint]
print("Y_train shape:", Y_train.shape, "dtype:", Y_train.dtype, "min:", np.min(Y_train), "max:", np.max(Y_train))
# Fit the network to the training dataset. The validation dataset can be used instead of a validataion split
history = model.fit(
    X_train, Y_train,
    validation_split=0.1,
    batch_size=BATCH,
    epochs=EPOCHS,
    callbacks=callbacks
)
'''
# Uncomment lines 84-85 and comment line 78 to run a previous model for prediction. Uncommenting lines 84-86 will 
# allow for training continuation in the event that the training was interuppted for whatever reason. If this is the 
# case, please comment out line 78 as well

latest_checkpoint = r'models_5/unet.23-0.89.hdf5'
model = tf.keras.models.load_model(
    latest_checkpoint,
    #custom_objects={'dice_coef_loss': dice_coef_loss}

)
model.summary()

'''
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='models_5_dice_loss/unet.{epoch:02d}-{f1:.2f}.hdf5',
    verbose=1,
    save_best_only=False
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    LearningRateScheduler(schedlr, verbose=1),
    checkpoint
]
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=43, callbacks=callbacks, initial_epoch=43)

if os.path.isdir(r'results_5_dice_loss') == True:
    shutil.rmtree('results_5_dice_loss')
'''
if os.path.isdir(r'results_5') == True:
    #os.mkdir('results_5_dice_loss')


    _, test_fol, _ = next(os.walk(test_dataset))
    # Assume all test folders have the same number of images, using the first folder to determine that number.
    _, _, test_files = next(os.walk(os.path.join(test_dataset, test_fol[0], 'image')))
    test_imgs = len(test_files)
    test_ids = list(range(1, test_imgs + 1))

    X_test = {}
    Y_test = {}


    # Create arrays for each test folder
    for folder in test_fol:
        X_test[folder] = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        Y_test[folder] = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
    
    # Load testing images and masks for each
    for folder in test_fol:
        test_path = os.path.join(test_dataset, folder)
        for n, id_ in tqdm(enumerate(test_files), total=len(test_files)):
            X_test[folder][n] = imread(os.path.join(test_path, 'image', f"{id_}"))
            mask = np.zeros((IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)
            for _ in next(os.walk(os.path.join(test_path, 'mask'))):
                mask_ = imread(os.path.join(test_path, 'mask', f"{id_}"))
                mask_ = np.expand_dims(mask_, axis=-1)
                mask = np.maximum(mask, mask_)
            Y_test[folder][n] = mask
'''
for i in test_fol:
    if os.path.isdir('results_5_dice_loss/%s' % i) == False:
        os.mkdir('results_5_dice_loss/%s' % i)

    save_dir = os.path.join('results_5_dice_loss', str(i))
    print(type(i), i)

    pred_test = model.predict(X_test[i], verbose=1)
    pred_test_mask = (pred_test > 0.4).astype(np.uint8)

    # for n in range(len(pred_test_mask)):
    #     outputmask = np.squeeze(pred_test_mask[n]*255)
    #     saveimg = Image.fromarray(outputmask, 'L')
    #     saveimg.save(os.path.join(save_dir, str(n)).replace('\\','/') + '.png', 'PNG')

    for n, fname in enumerate(test_files):
      outputmask = np.squeeze(pred_test_mask[n] * 255)
      saveimg = Image.fromarray(outputmask.astype(np.uint8), 'L')
      saveimg.save(os.path.join(save_dir, fname), 'PNG')

for i in test_fol:
    results_dir = os.path.join('results_5_dice_loss', str(i))
    imgstitch(results_dir)
'''

print("\nEvaluating model on the test set...")

 
# Flatten the dict of test arrays to single X_test_flat, Y_test_flat
X_test_flat = np.concatenate([X_test[k] for k in X_test], axis=0)
Y_test_flat = np.concatenate([Y_test[k] for k in Y_test], axis=0)
# Evaluate using the model (assuming it was compiled with desired metrics)
results = model.evaluate(X_test_flat, Y_test_flat, batch_size=8, verbose=1)

# Print nicely
metric_names = model.metrics_names
for name, value in zip(metric_names, results):
    print(f"{name}: {value:.4f}")

all_precisions = []
all_recalls = []
thresholds = np.linspace(0.1, 0.9, 9)

for folder in test_fol:
    print(f"Evaluating relaxed metrics on test folder: {folder}")
    
    preds = model.predict(X_test[folder], verbose=1)
    for th in thresholds:
        preds_binary = (preds > th).astype(np.uint8)

        for i in range(len(preds_binary)):
            pred_mask = np.squeeze(preds_binary[i])     # Shape: (H, W)
            true_mask = np.squeeze(Y_test[folder][i])    # Shape: (H, W)

            prec, rec = relaxed_metrics(pred_mask, true_mask, slack=3)
            all_precisions.append(prec)
            all_recalls.append(rec)
import numpy as np

# Sample relaxed precision and recall values at different thresholds


all_precisions = np.array(all_precisions)
all_recalls = np.array(all_recalls)

diff = np.abs(all_precisions - all_recalls)  # ✅ now this works
# Find index where difference is minimized (closest to Precision = Recall)
bep_index = np.argmin(diff)

# Extract break-even point
bep_precision = all_precisions[bep_index]
bep_recall = all_recalls[bep_index]

print(f"Break-Even Point: Precision = Recall = {bep_precision:.4f}")
# Final averaged scores
avg_relaxed_precision = np.mean(all_precisions)
avg_relaxed_recall = np.mean(all_recalls)

print(f"\nRelaxed Precision (ρ=3): {avg_relaxed_precision:.4f}")
print(f"Relaxed Recall (ρ=3): {avg_relaxed_recall:.4f}")

import matplotlib.pyplot as plt

# Assume 'history' is your model training history object
# Extract values
train_loss = history.history.get('loss') or history.history.get('dice_coef_loss')
val_loss = history.history.get('val_loss') or history.history.get('val_dice_coef_loss')
train_f1 = history.history.get('f1') or history.history.get('f1_score')
val_f1 = history.history.get('val_f1') or history.history.get('val_f1_score')

# Check available data before plotting
if train_loss and val_loss:
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot F1 Score
    if train_f1 and val_f1:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_f1, 'b-', label='Train F1')
        plt.plot(epochs, val_f1, 'r-', label='Val F1')
        plt.title('F1 Score over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

    plt.tight_layout()
    plt.savefig('loss_and_f1_dice_loss.png')
    plt.show()
else:
    print("Training loss or validation loss not found in history.")
    print("Available keys:", history.history.keys())
    
epochs = range(1, len(history.history['loss']) + 1)
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_f1 = history.history.get('f1_score')  # or another key depending on your metric
val_f1 = history.history.get('val_f1_score')  # same

# Plot Loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b-', label='Train Loss')
plt.plot(epochs, val_loss, 'r-', label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot F1 Score
plt.subplot(1, 2, 2)
plt.plot(epochs, train_f1, 'b-', label='Train F1')
plt.plot(epochs, val_f1, 'r-', label='Val F1')
plt.title('F1 Score over Epochs')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()

plt.tight_layout()
plt.savefig('loss_and_f1_dice_loss.png')
plt.show()