# -*- coding: utf-8 -*-
"""
@avanc
This file is for various tested U-Net and ResUNet 
variations used during development and ablation. See model_resunt
and model_unet for the final model used in experiments.
To use this, comment out unused models, and compile the correct one in train.py
"""

################################## LIBRARIES ##################################

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, concatenate, Lambda, UpSampling2D
from tensorflow.keras import Model, Input
from tensorflow.keras import layers, models
from contextlib import redirect_stdout
import tensorflow as tf
import tensorflow_addons as tfa

####################################################################
# BASELINE U-NET MODEL variants 
'''these are basic U-Net variants used for quick testing and debugging.
- SimpleUNet: really shallow, used to just check that model runs.
- MidUNet: adds one encoder-decoder level.
- UNetV3: proper U-Net with skip connections and two decoder stages, similar to actual paper.
'''
def SimpleUNet(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(x)

    return models.Model(inputs, x)

def MidUNet(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)

    # Bottleneck
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    # Decoder
    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(x)

    return models.Model(inputs, outputs)


def UNetV3(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(128, 3, activation='relu', padding='same')(b)

    # Decoder
    u2 = layers.UpSampling2D(size=(2, 2))(b)
    u2 = layers.Concatenate()([u2, c2])
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)

    u1 = layers.UpSampling2D(size=(2, 2))(c3)
    u1 = layers.Concatenate()([u1, c1])
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = layers.Conv2D(32, 3, activation='relu', padding='same')(c4)

    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c4)

    return models.Model(inputs=inputs, outputs=outputs)
####################################################################
# RESUNET model variants
'''
ResUNetV4: Stripped-down ResUNet with dropout and no norm. Helped debug NaNs.
ResUNetV5: Deeper version of V4; used to test overfitting/generalization.
Both clip the sigmoid output to avoid log(0) during BCE.
safe_residual_block:
-used to debug early issues ( NaNs) by stripping normalization
and residual addition. Helped isolate whether instability was caused by BatchNorm,
GroupNorm, or the residual connection.
'''
# tested this safe res block to test if NaNs were due to Normlization or residual connection
def safe_residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='glorot_uniform')(x)
    #x = tfa.layers.GroupNormalization(groups=8)(x)  # 8 groups is a common choice
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='glorot_uniform')(x)
    #x = tfa.layers.GroupNormalization(groups=8)(x)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        #shortcut = tfa.layers.GroupNormalization(groups=8)(shortcut)

    # x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x
    #dont have resiudal temporarily

def ResUNetV4(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = safe_residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(e1)

    e2 = safe_residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(e2)

    # bottleneck
    b = safe_residual_block(p2, 128)

    # decoder
    u2 = layers.UpSampling2D()(b)
    u2 = layers.concatenate([u2, e2])
    d2 = safe_residual_block(u2, 64)
    d2 = tf.keras.layers.Dropout(0.3)(d2)  # #dropout after conv block

    u1 = layers.UpSampling2D()(d2)
    u1 = layers.concatenate([u1, e1])
    d1 = safe_residual_block(u1, 32)
    d1 = tf.keras.layers.Dropout(0.3)(d1)

    # output
    #outputs = layers.Conv2D(1, 1, activation='sigmoid')(d1)
    outputs = layers.Conv2D(1, 1)(d1)
    #outputs = tf.keras.layers.Activation(lambda x: tf.clip_by_value(tf.nn.sigmoid(x), 1e-7, 1 - 1e-7))(outputs)
    outputs = layers.Activation('sigmoid')(outputs)
    outputs = layers.Lambda(lambda x: tf.clip_by_value(x, 1e-5, 1 - 1e-5))(outputs)
    #clipping ensures model never outputs exactly 0 or 1 after sigmoid

    return models.Model(inputs, outputs)

#trying somehting a bit deeper
def ResUNetV5(input_shape=(224, 224, 3)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    e1 = safe_residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(e1)

    e2 = safe_residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(e2)

    e3 = safe_residual_block(p2, 128)
    p3 = layers.MaxPooling2D()(e3)

    # Bottleneck
    b = safe_residual_block(p3, 256)

    # Decoder
    u3 = layers.UpSampling2D()(b)
    u3 = layers.concatenate([u3, e3])
    d3 = safe_residual_block(u3, 128)

    u2 = layers.UpSampling2D()(d3)
    u2 = layers.concatenate([u2, e2])
    d2 = safe_residual_block(u2, 64)

    u1 = layers.UpSampling2D()(d2)
    u1 = layers.concatenate([u1, e1])
    d1 = safe_residual_block(u1, 32)

    # Output
    outputs = layers.Conv2D(1, 1)(d1)
    outputs = layers.Activation('sigmoid')(outputs)
    outputs = layers.Lambda(lambda x: tf.clip_by_value(x, 1e-5, 1 - 1e-5))(outputs)

    return models.Model(inputs, outputs)


###########################################DEEP GROUPNORM RESUNET ###########################################
'''
another variant I tried with groupNorm so we could use this instead of BN
'''
def group_norm_dynamic(x):
    channels = x.shape[-1]
    #print("GroupNorm input stats - min:", tf.reduce_min(x).numpy(), "max:", tf.reduce_max(x).numpy(), "mean:", tf.reduce_mean(x).numpy())
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            groups = g
            break
    return tfa.layers.GroupNormalization(
        groups=groups,
        epsilon=1e-3,
        gamma_initializer='ones',
        beta_initializer='zeros'
    )(x)

def residual_block(x, filters, stride=(1,1)):
    shortcut = x

    #tested with BN and group norm
    x = BatchNormalization()(x)
    #x = group_norm_dynamic(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), strides=stride, padding='same')(x)

    x = BatchNormalization()(x)
    #x = tfa.layers.GroupNormalization(groups=8)(x)
    #x = group_norm_dynamic(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (3,3), strides=(1,1), padding='same')(x)

    # adjust shortcut if needed
    if shortcut.shape[-1] != filters or stride != (1,1):
        shortcut = Conv2D(filters, (1,1), strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
        #shortcut = tfa.layers.GroupNormalization(groups=8)(shortcut)
        #shortcut = group_norm_dynamic(shortcut)

    x = Add()([shortcut, x])
    return x

# # ###########################################
# Encoder
###########################################
def encoder(x):
    skips = []

    # Level1
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    skips.append(x)

    # Leve 2
    x = residual_block(x, 128, stride=(2,2))
    x = residual_block(x, 128)
    skips.append(x)

    # Level3
    x = residual_block(x, 256, stride=(2,2))
    x = residual_block(x, 256)
    skips.append(x)

    # Level4
    x = residual_block(x, 512, stride=(2,2))
    x = residual_block(x, 512)
    skips.append(x)

    return x, skips

# # ###########################################
# Decoder
###########################################
def decoder(x, skips):
    # Reverse skip connections
    skips = skips[::-1]

    # l1
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = concatenate([x, skips[0]])
    x = residual_block(x, 256)

    # 2
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = concatenate([x, skips[1]])
    x = residual_block(x, 128)

    # 3
    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = concatenate([x, skips[2]])
    x = residual_block(x, 64)

    return x

#############################################
# Full Deep ResUNet Model
###########################################
def build_deep_resunet(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)

    # encoder
    x, skips = encoder(inputs)

    # Bridge
    x = residual_block(x, 512, stride=(2,2))
    x = residual_block(x, 512)

    # decoder
    x = decoder(x, skips)

    x = UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = Conv2D(64, (3,3), padding='same', activation='relu')(x)

    # Output Layer
    outputs = Conv2D(1, (1,1), activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model



## example 

# model = ResUNet((224, 224, 3))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
# tf.keras.utils.plot_model(model, to_file='model.png', show_layer_names=True, show_shapes=True, rankdir='TB')


