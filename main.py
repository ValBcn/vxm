# imports
import os, sys

sys.path.append('/home/valentin/local_packages/keras_med_io/')

import numpy as np
import cv2
import argparse
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import interpolation
from scripts.resize import resize_data
import nibabel as nib
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports
import voxelmorph as vxm
import neurite as ne
from scripts.vxm_gen import vxm_data_generator

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="For training on Fetal MRI Segmentation using Capsule Networks and CNNs")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")
    parser.add_argument("--img_path1", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped images are")
    parser.add_argument("--img_path2", type = str, required = False,
                        help = "If bi-model registration. Path to the base directory where the preprocessed/reshaped images 2 are")
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs")
    parser.add_argument("--batch_size", type = int, required = False, default = 2,
                        help = "Batch size for the CNN should be 17 and for the Capsule Network, it should be 2.")
    parser.add_argument("--lr", type = float, required = False, default = 1e-4,
                        help = "The learning rate")
    parser.add_argument("--trainset", type = float, required = False, default = .6,
                        help = "Size of the training set.")
    parser.add_argument("--valset", type = float, required = False, default = .2,
                        help = "Size of the validation set.")
    parser.add_argument("--testset", type = float, required = False, default = .2,
                        help = "Size of the testing set.")
    parser.add_argument("--multi", type = boolean_string, required = False, default = False,
                        help = "multi or uni modal registration.")


    args = parser.parse_args()

    # loading data

    img_path1 = os.path.join(args.img_path1)
    img_path2 = os.path.join(args.img_path2)
    data = []
    data2 = []

    # Resizing MRI
    for img in os.listdir(img_path1):
        img_array = np.load(os.path.join(img_path1,img))
        data.append(img_array)
    data = np.stack(data)

    data = np.asarray(data)
    
    x_train, x_val = train_test_split(data, test_size=0.2)

    vol_shape = x_train.shape[1:]
    print('train shape', x_train.shape)
    print('val shape', x_val.shape)
    print('vol_shape',vol_shape)


    if args.multi is True:
        # Resizing US
        for img in os.listdir(img_path2):
            img_array2 = np.load(os.path.join(img_path2,img))
            data2.append(img_array2)
        data2 = np.stack(data2)

        data2 = np.asarray(data2)
    
        x_train2, x_val2 = train_test_split(data2, test_size=0.2)

        vol_shape2 = x_train2.shape[1:]
        print('train_2 shape', x_train2.shape)
        print('val_2 shape', x_val2.shape)
        print('vol_shape_2',vol_shape2)

    
    # configure unet input shape (concatenation of moving and fixed images)
    ndim = 3
    unet_input_features = 2
    inshape = (*x_train.shape[1:], unet_input_features)

    # configure unet features 
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]



    # unet
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    # losses and loss weights
    losses = ['mse', vxm.losses.Grad('l2').loss]
    loss_weights = [1, 0.01]

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=loss_weights)

    if args.multi == True:
        train_generator = vxm_data_generator(x_train,x_train2, batch_size=2)
    if args.multi == False:
        train_generator = vxm_data_generator(x_train,x_train, batch_size=2)

    in_sample, out_sample = next(train_generator)

    hist = vxm_model.fit(train_generator, epochs=args.epochs, steps_per_epoch=5, verbose=2)

    if args.multi == True:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5"))
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5")))
    if args.multi == False:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5"))
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5")))
