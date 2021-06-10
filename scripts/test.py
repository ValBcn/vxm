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

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="For training on Fetal MRI Segmentation using Capsule Networks and CNNs")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")                                                                                                           
    parser.add_argument("--input1", type = str, required = True,
                        help = "Path to the first input")
    parser.add_argument("--input2", type = str, required = True,
                        help = "Path to the second input")
    parser.add_argument("--input2_seg", type = str, required = False,
                        help = "Path to the second input segmentation which will be warped using the predicted registration field")
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs")
    parser.add_argument("--lr", type = float, required = False, default = 1e-4,
                        help = "The learning rate")
    parser.add_argument("--res_dir", type = str, required = True,
                        help = "Path to the directory where the results are saved")
    parser.add_argument("--multi", type = boolean_string, required = False, default = False,
                        help = "multi or uni modal registration.")
    parser.add_argument("--reshape", type = int, required = False, default = 64,
                        help = "size x of the reshaping (new shape = (x,x,x).")

    
    args = parser.parse_args()

    img_array = np.load(args.input1,allow_pickle=True)
    img_array = resize_data(img_array,int(args.reshape/2),args.reshape,args.reshape)
    vol_shape = img_array.shape
    print ('vol shape = ', vol_shape)
    img_array = np.reshape(img_array, (1,) + img_array.shape + (1,))

    if args.multi == True:
        img_array2 = np.load(args.input2,allow_pickle=True)[0]
    if args.multi == False:
        img_array2 = np.load(args.input2)
    img_array2 = resize_data(img_array2,int(args.reshape/2),args.reshape,args.reshape)
    
    img_array2 = np.reshape(img_array2, (1,) + img_array2.shape + (1,))

    
    # model

    # configure unet features                                                                                                                                                                               
    nb_features = [
        [32, 32, 32, 32],         # encoder features                                                                                                                                                        
        [32, 32, 32, 32, 32, 16]  # decoder features                                                                                                                                                        
    ]

    # unet                                                                                                                                                                                                  
    vxm_model = vxm.networks.VxmDense(vol_shape, nb_features, int_steps=0)

    if args.multi == False:
        weights = vxm_model.load_weights(os.path.join(args.weights_dir,  str(args.epochs) + "epochs_uni.h5"))
    if args.multi == True:
        weights = vxm_model.load_weights(os.path.join(args.weights_dir,  str(args.epochs) + "epochs_multi.h5"))
    
    # prediction

    val_input = [img_array2,img_array] # moving and fixed, in that order
    val_pred = vxm_model.predict(val_input)

    flow = val_pred[1].squeeze()

    # warping the segmentation

    #np.load(args.input2_seg)

    print (val_pred[0].shape)
    for i in range (len(val_input)):
        print(val_input[i].shape)
        #val_input[i] = np.squeeze(val_input[i])
        #val_input[i] = np.reshape(val_input[i], (1,) + val_input[i].shape + (1,))
        print(val_input[i].shape)
    
    imgs = np.asarray([img[0, :, :, :, 0] for img in val_input + val_pred])

    
    moving_nii = nib.Nifti1Image(imgs[0,:,:,:], np.eye(4))
    fixed_nii = nib.Nifti1Image(imgs[1,:,:,:], np.eye(4))
    warped_nii = nib.Nifti1Image(imgs[2,:,:,:], np.eye(4))

    print('saving images in :' , args.res_dir )

    if args.multi == False:
        nib.save(fixed_nii, os.path.join(args.res_dir, str(args.epochs) + '_fixed_uni.nii'))
        nib.save(moving_nii, os.path.join(args.res_dir, str(args.epochs) + '_moving_uni.nii'))
        nib.save(warped_nii, os.path.join(args.res_dir, str(args.epochs) + '_warped_uni.nii'))
        np.save(os.path.join(args.res_dir, str(args.epochs) + '_deffield_uni.npy'),imgs[3,:,:,:])
        np.save(os.path.join(args.res_dir, str(args.epochs) + '_flow_uni.npy'),flow)
    if args.multi == True:
        nib.save(fixed_nii, os.path.join(args.res_dir, str(args.epochs) + '_fixed_multi.nii'))
        nib.save(moving_nii, os.path.join(args.res_dir, str(args.epochs) + '_moving_multi.nii'))
        nib.save(warped_nii, os.path.join(args.res_dir, str(args.epochs) + '_warped_multi.nii'))
        np.save(os.path.join(args.res_dir, str(args.epochs) + '_deffield_multi.npy'),imgs[3,:,:,:])
        np.save(os.path.join(args.res_dir, str(args.epochs) + '_flow_multi.npy'),flow)
    
   
