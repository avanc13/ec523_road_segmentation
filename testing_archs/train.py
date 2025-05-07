# -*- coding: utf-8 -*-
"""
Various training visualizations and testing--use train.py in various unet/resunet folders for 
to train the final models we used. This is for debugging***
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import model_resunet
import numpy as np
import tensorflow as tf
from math import floor
from tqdm import tqdm
from skimage.io import imread
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import AUC

import random
import scipy.misc
from PIL import Image
import shutil
from utils import imgstitch, DatasetLoad

########################### LEARNING RATE SCHEDULER ###########################
tf.keras.backend.clear_session()
# Function for learning rate decay. The learning rate will reduce by a factor of 0.1 every 10 epochs.
# def schedlr(epoch, lr):
#     new_lr = 0.001 * (0.1)**(floor(epoch/10))
#     return new_lr

############################### HYPERPARAMETERS ###############################

IMG_SIZE = 224
BATCH = 16
EPOCHS = 15 #100
print("Model is being initialized from scratch.")

################################### DATASET ###################################

# Paths for relevant datasets to load in
train_dataset = r'dataset/samples_train'
test_dataset = r'dataset/test_patch'
#val_dataset = r'dataset/samples_val'

# Make a list of the test folders to be used when predicting the model. This will be fed into the prediction
# flow to generate the stitched image based off the predictions of the patches fed into the network
_, test_fol, _ = next(os.walk(test_dataset))

# Load in the relevant datasets 
X_train, Y_train, X_test, Y_test, X_val, Y_val = DatasetLoad(train_dataset, test_dataset, val_dataset=None, max_train_samples=100)
# Check for NaNs/Infs in X
print("X_train NaNs:", np.isnan(X_train).any())
print("X_train Infs:", np.isinf(X_train).any())
print("X_train range:", np.min(X_train), np.max(X_train))

# Check for NaNs/Infs in Y
print("Y_train NaNs:", np.isnan(Y_train).any())
print("Y_train Infs:", np.isinf(Y_train).any())
print("Y_train range:", np.min(Y_train), np.max(Y_train))

# Check if Y is binary
unique_labels = np.unique(Y_train)
print("Unique values in Y_train:", unique_labels)
if not np.all(np.isin(unique_labels, [0, 1])):
    print("warning: Y_train contains values other than 0 and 1!")

# Class imbalance check
print("Proportion of foreground pixels (label=1):", np.mean(Y_train))

# idx = random.randint(0, len(X_train) - 1)

# # Get the corresponding image and mask
# image = X_train[idx]
# mask = Y_train[idx].squeeze()

# # Plot image and mask side-by-side
# plt.figure(figsize=(10, 4))

# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title(f"Input Image (idx: {idx})")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(mask, cmap='gray')
# plt.title(f"Ground Truth Mask (idx: {idx})")
# plt.axis('off')

# plt.tight_layout()
# plt.show()

################################ RESIDUAL UNET ################################

sgd_optimizer = Adam(learning_rate=1e-4, clipnorm=1.0)


# Metrics to be used when evaluating the network
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

#f1 = F1Score(num_classes=1, name='f1', average='micro', threshold=0.4)
# For ROC AUC (default)
#auc_roc = AUC(name='roc_auc')

# For Precision-Recall AUC
#auc_pr = AUC(name='pr_auc', curve='PR')

# Instantiate the network 

#model = model_resunet.ResUNet((IMG_SIZE, IMG_SIZE, 3))
model = model_resunet.build_deep_resunet((IMG_SIZE, IMG_SIZE, 3))
class DebugLoss(tf.keras.losses.Loss):
    def __init__(self, base_loss):
        super().__init__()
        self.base_loss = base_loss

    def call(self, y_true, y_pred):
        tf.debugging.check_numerics(y_pred, "NaNs in y_pred in loss")
        tf.debugging.check_numerics(y_true, "NaNs in y_true in loss")
        return self.base_loss(y_true, y_pred)
#added this for very imbalanced dataset
def focal_loss(gamma=2.0, alpha=0.25):
    def loss_fn(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma) * y_true + (1 - alpha) * tf.pow(y_pred, gamma) * (1 - y_true)
        return tf.reduce_mean(weight * cross_entropy)
    return loss_fn
#that one didnt work lets try this with a huge weight
def weighted_bce(pos_weight=20.0, neg_weight=1.0):
    def loss_fn(y_true, y_pred):
        #epsilon = tf.keras.backend.epsilon()
        epsilon = 1e-5  # or even 1e-4 if needed
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        loss = -(pos_weight * y_true * tf.math.log(y_pred) +
                 neg_weight * (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(loss)
    return loss_fn

def safe_binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1. - 1e-6)
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# comment the ones we dont wanna use--testing

#loss_fn = DebugLoss(weighted_bce(pos_weight=10.0))
#loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
loss_fn = weighted_bce(pos_weight=20.0, neg_weight=1.0)


#Manual print of batch 0 before training starts jusy in case
def run_debug_pass(model, X_batch):
    print("\n=== Debug pass ===")
    preds = model(X_batch, training=True)

    # Print simple stats outside of TF graph
    print("Predictions:")
    print(f"  min = {np.min(preds.numpy()):.6f}")
    print(f"  max = {np.max(preds.numpy()):.6f}")
    print(f"  mean = {np.mean(preds.numpy()):.6f}")
    print(f"  stddev = {np.std(preds.numpy()):.6f}")

    if np.isnan(preds.numpy()).any():
        print(" NaNs detected in predictions!")

    return preds
X_batch = X_train[:8]
run_debug_pass(model, X_batch)

x0 = X_train[:BATCH]
y0 = Y_train[:BATCH]
y0_pred = model(x0, training=False)
print("[Pre-training check] Batch 0 y_pred stats:")
print("min:", tf.reduce_min(y0_pred).numpy())
print("max:", tf.reduce_max(y0_pred).numpy())
print("mean:", tf.reduce_mean(y0_pred).numpy())

#manually run an update to see where it all blows up
@tf.function
def train_step_debug(model, x, y, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        tf.debugging.check_numerics(y_pred, "NaNs in y_pred(before loss)")
        loss = loss_fn(y, y_pred)
    grads = tape.gradient(loss, model.trainable_weights)
    grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for i, grad in enumerate(grads):
        tf.debugging.check_numerics(grad, f"NaNs in gradient {i}")
        if tf.reduce_any(tf.math.is_nan(grad)):
            print(f"NaNs in gradient {i}: {model.trainable_variables[i].name}")
    return loss

for i in range(5):
    x_batch = X_train[i*BATCH:(i+1)*BATCH]
    y_batch = Y_train[i*BATCH:(i+1)*BATCH]
    print(f"\n== Debugging batch {i} → X_train[{i*BATCH}:{(i+1)*BATCH}] ===")
    try:
        loss_val = train_step_debug(model, x_batch, y_batch, loss_fn, sgd_optimizer)
        print(f" Batch {i} passed! Loss: {loss_val.numpy():.4f}")
    except tf.errors.InvalidArgumentError as e:
        print(f"NaNs in batch {i}: {e.message}")


# Run one gradient update manually on a small batch

# print("Running focused manual debug step...")
# loss_val = train_step_debug(model, X_train[:8], Y_train[:8], loss_fn, optimizer=sgd_optimizer)
# print("Manual step completed. Loss =", loss_val.numpy())



model.compile(optimizer=sgd_optimizer, loss=loss_fn, metrics=['accuracy', precision, recall])
model.summary()

# Callacks to be used in the network
#checkpoint_path = os.path.join(dname, 'models', 'resunet.{epoch:02d}-{f1:.2f}.hdf5')
checkpoint_path = os.path.join(dname, 'models', 'resunet.{epoch:02d}-{val_loss:.4f}.hdf5')

checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=False)

callbacks =[
    tf.keras.callbacks.EarlyStopping(patience=7, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    #LearningRateScheduler(schedlr, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    checkpoint]

class CheckNumericsCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        try:
            x_batch = X_train[batch * BATCH : (batch + 1) * BATCH]
            y_pred = self.model(x_batch, training=False)
            tf.debugging.check_numerics(y_pred, message=f"NaNs in prediction at batch {batch}")
        except tf.errors.InvalidArgumentError as e:
            print(f" NaN detected in prediction at batch {batch}:\n{e.message}\n")
            self.model.stop_training = True

callbacks.append(CheckNumericsCallback())
#helps you see prediction masks
class PredictionVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, sample_input, sample_ground_truth):
        self.sample_input = sample_input
        self.sample_ground_truth = sample_ground_truth

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model(self.sample_input, training=False).numpy()
        thresholded = (pred > 0.4).astype(np.uint8)
        print("Raw prediction min:", pred.min())
        print("Raw prediction max:", pred.max())


        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Epoch {epoch+1} Predictions")

        plt.subplot(1, 3, 1)
        plt.imshow(pred[0].squeeze(), cmap='gray')
        plt.title("Raw Prediction (gray)")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred[0].squeeze(), cmap='hot')
        plt.title("Raw Prediction (heatmap)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(thresholded[0].squeeze(), cmap='gray')
        plt.title("Binary Mask (>0.4)")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
sample_input = X_train[:1]
sample_ground_truth = Y_train[:1]
callbacks.append(PredictionVisualizer(sample_input, sample_ground_truth))



# Fit the network to the training dataset. The validation dataset can be used instead of a validataion split
print("Label counts in batch 0:")
unique, counts = np.unique(Y_train[:BATCH], return_counts=True)
for u, c in zip(unique, counts):
    print(f"Value: {u}, Count: {c}, Ratio: {c / np.prod(Y_train[:BATCH].shape):.4f}")
history = model.fit(X_train, Y_train, validation_split=0.1, batch_size=BATCH, epochs=EPOCHS, callbacks=callbacks)

########################### PREDICTION AND RESULTS ############################

# If previous results exist, delete them so the results won't be mixed up
if os.path.isdir(r'results') == True:
    shutil.rmtree('results')

# Make new results directory along with sub directories for each of the test images
if os.path.isdir(r'results') == False:
    os.mkdir('results')

# Generate the predicted masks for each of the test images and save the patches for use when restitching the image
for i in test_fol:    
    if os.path.isdir('results/%s' % i) == False:
        os.mkdir('results/%s' % i)
    
    save_dir = os.path.join('results', str(i))
    assert np.isfinite(X_test[i]).all(), f"NaNs/Infs in X_test for folder {i}"

    pred_test = model.predict(X_test[i], verbose=1)
    pred_test = np.clip(pred_test, 1e-5, 1 - 1e-5)

    assert np.isfinite(pred_test).all(), f"NaNs/Infs in predictions for folder {i}"
    #pred_test_mask = (pred_test > 0.4).astype(np.uint8)
    print("NaNs in pred_test:", np.isnan(pred_test).any())
    print("Infs in pred_test:", np.isinf(pred_test).any())
    ###code to visulaize 

    # Visualize one of the raw predictions
    n = 0  # change index if needed
    print(f" Predictions for test folder: {i}, patch {n}")
    print("Raw prediction stats:", 
          "min:", np.min(pred_test[n]), 
          "max:", np.max(pred_test[n]), 
          "mean:", np.mean(pred_test[n]))

    pred_test_mask = (pred_test > 0.4).astype(np.uint8)
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(pred_test[n].squeeze(), cmap='gray')
    plt.title("Raw Prediction (gray)")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred_test[n].squeeze(), cmap='hot', vmin=0, vmax=1)
    plt.title("Raw Prediction (heatmap)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_test_mask[n].squeeze(), cmap='gray')
    plt.title("Binary Mask (>0.4)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    for n in range(len(pred_test_mask)):
        outputmask = np.squeeze(pred_test_mask[n]*255)
        saveimg = Image.fromarray(outputmask, 'L')
        saveimg.save(os.path.join(save_dir, str(n)).replace('\\','/') + '.png', 'PNG')

# Loop through the entire test prediction dataset and feed the images as an input to the stitching function
# for i in test_fol: 
#     results_dir = os.path.join('results', str(i))
#     imgstitch(results_dir)

#plotting, ignoring stitiching for now

def plot_training_curves(history):
    """
    Plots training and validation loss (and accuracy if available) from a Keras history object.
    """
    # Loss
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Accuracy (optional if available)
    if 'accuracy' in history.history:
        plt.figure(figsize=(10,5))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history.history:
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Precision and Recall if you tracked them
    if 'precision' in history.history:
        plt.figure(figsize=(10,5))
        plt.plot(history.history['precision'], label='Train Precision')
        if 'val_precision' in history.history:
            plt.plot(history.history['val_precision'], label='Validation Precision')
        plt.title('Precision Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        plt.show()

    if 'recall' in history.history:
        plt.figure(figsize=(10,5))
        plt.plot(history.history['recall'], label='Train Recall')
        if 'val_recall' in history.history:
            plt.plot(history.history['val_recall'], label='Validation Recall')
        plt.title('Recall Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        plt.show()

plot_training_curves(history)
