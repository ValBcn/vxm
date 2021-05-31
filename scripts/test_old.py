# imports                                                                                                                                                                                                   
import os, sys

sys.path.append('/home/valentin/local_packages/keras_med_io/')

import numpy as np
import cv2
import argparse
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import train_test_split
import nibabel as nib
from resize import resize_data
assert tf.__version__.startswith('2.'), 'This tutorial assumes Tensorflow 2.0+'

# local imports                                                                                                                                                                                             
import voxelmorph as vxm
import neurite as ne
from vxm_gen import vxm_data_generator

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="For training on Fetal MRI Segmentation using Capsule Networks and CNNs")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")                                                                                                           
    parser.add_argument("--dset_path", type = str, required = True,
                        help = "Path to the base directory where the preprocessed imagesTr and labelsTr directory are.")
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs")
    parser.add_argument("--lr", type = float, required = False, default = 1e-4,
                        help = "The learning rate")
    parser.add_argument("--res_dir", type = str, required = True,
                        help = "Path to the directory where the results are saved")

    args = parser.parse_args()

    img_path = os.path.join(args.dset_path,"imagesTr")
    data = []

    for img in os.listdir(img_path):
        img_array = np.load(os.path.join(img_path,img))
        img_array = resize_data(img_array,64,64,64)
        data.append(img_array)
    data = np.stack(data)

    data = np.asarray(data)

    x_train, x_val = train_test_split(data, test_size=0.2)

    vol_shape = x_train.shape[1:]
    print('train shape', x_train.shape)
    print('val shape', x_val.shape)
    print('vol_shape',vol_shape)

    # model

    # configure unet features                                                                                                                                                                               
    nb_features = [
        [32, 32, 32, 32],         # encoder features                                                                                                                                                        
        [32, 32, 32, 32, 32, 16]  # decoder features                                                                                                                                                        
    ]

    # unet                                                                                                                                                                                                  
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    weights = vxm_model.load_weights(os.path.join(args.weights_dir,  str(args.epochs) + "epochs.h5"))
    
    # create the validation data generator
    val_generator = vxm_data_generator(x_val, x_val, batch_size = 1)
    val_input, _ = next(val_generator)
    val_input2, _ = next(val_generator)

    print ('val_input shape', val_input[1].shape)
    # prediction
    val_pred = vxm_model.predict(val_input)
    
    imgs = np.asarray([img[0, :, :, :, 0] for img in val_input + val_pred])

    fixed_nii = nib.Nifti1Image(imgs[0,:,:,:], np.eye(4))
    moving_nii = nib.Nifti1Image(imgs[1,:,:,:], np.eye(4))
    warped_nii = nib.Nifti1Image(imgs[2,:,:,:], np.eye(4))

    nib.save(fixed_nii, os.path.join(args.res_dir, str(args.epochs) + '_fixed.nii'))
    nib.save(moving_nii, os.path.join(args.res_dir, str(args.epochs) + '_moving.nii'))
    nib.save(warped_nii, os.path.join(args.res_dir, str(args.epochs) + '_warped.nii'))

    np.save(os.path.join(args.res_dir, str(args.epochs) + '_fixed.npy'),imgs[0,:,:,:])
    
