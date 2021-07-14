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
from itertools import combinations, permutations
from scipy.interpolate import interpn
from sklearn.metrics import jaccard_score


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
    parser.add_argument("--img_dir", type = str, required = True,
                        help = "Path to the image directory")
    parser.add_argument("--lab_dir", type = str, required = True,
                        help = "Path to the label directory")
    parser.add_argument("--epochs", type = int, required = True,
                        help = "Number of epochs used for training")
    parser.add_argument("--res_dir", type = str, required = True,
                        help = "Path to the directory where the results are saved")
    parser.add_argument("--multi", type = boolean_string, required = False, default = False,
                        help = "multi or uni modal registration.")
    parser.add_argument("--reshape", type = int, required = False, default = 64,
                        help = "size x of the reshaping used for training (new shape = (x,x,x)).")

    
    args = parser.parse_args()

    data_img = []
    data_lab = []

    for img in sorted(os.listdir(args.img_dir)):
        img_array = np.load(os.path.join(args.img_dir,img))
        img_array = np.reshape(img_array, (1,) + img_array.shape + (1,))
        data_img.append(img_array)
        
    for lab in sorted(os.listdir(args.lab_dir)):
        lab_array = np.load(os.path.join(args.lab_dir,lab)) # labels have the same name as images
        data_lab.append(lab_array)

    imgs = np.stack(data_img)
    imgs = np.asarray(imgs)
    img_pairs = list(combinations(imgs,2))
    labs = np.stack(data_lab)
    labs = np.asarray(labs)
    lab_pairs = list(combinations(labs,2))

    vol_shape = img_array.squeeze().shape
    print ('vol shape = ', vol_shape)
    
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

    
    xx = np.arange(vol_shape[1])
    yy = np.arange(vol_shape[0])
    zz = np.arange(vol_shape[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

        
    # prediction

    dices_before = []
    dices_after = []
    jacs_before = []
    jacs_after = []

    for i in range(len(img_pairs)):
        val_input = [img_pairs[i][0],img_pairs[i][1]] # moving and fixed, in that order
        val_pred = vxm_model.predict(val_input)
        flow = val_pred[1].squeeze()

        imgs = np.asarray([img[0, :, :, :, 0] for img in val_input + val_pred])

        moving = imgs[0,:,:,:]
        fixed = imgs[1,:,:,:]
        warped = imgs[2,:,:,:]
        labels_fixed = lab_pairs[i][1]
        labels_moving = lab_pairs[i][0]

        print('img size =', img_pairs[i][0].shape)
        print('lab size =', labels_moving.shape)

        label_warped = np.zeros(labels_moving.shape)

        sample = np.zeros(flow.shape)
        sample[:,:,:,0] = flow[:,:,:,1] + grid[:,:,:,0]
        sample[:,:,:,1] = flow[:,:,:,0] + grid[:,:,:,1]
        sample[:,:,:,2] = flow[:,:,:,2] + grid[:,:,:,2]
        
        sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)
        warped_label = interpn((yy, xx, zz), labels_moving, sample, method='nearest', bounds_error=False, fill_value=0)

        #warped_img = interpn((yy, xx, zz), fixed_np, sample, method='nearest', bounds_error=False, fill_value=0)

        dice_before = np.sum(labels_moving[labels_fixed==1])*2.0 / (np.sum(labels_fixed) + np.sum(labels_moving))
        dice_after = np.sum(warped_label[labels_fixed==1])*2.0 / (np.sum(labels_fixed) + np.sum(warped_label))

        dices_before.append(dice_before)
        dices_after.append(dice_after)

        #jac_before = jaccard_score(labels_moving.flatten(),labels_fixed.flatten())
        #jac_after = jaccard_score(warped_label.flatten(),labels_fixed.flatten())

        #jacs_before.append(jac_before)
        #jacs_after.append(jac_after)

        print('%i pair ================> \n' %(i))
        print('dice before = %f \n' %(dice_before))
        print('dice after = %f \n' %(dice_after))
        #print('jaccard before = %f \n' %(jac_before))
        #print('jaccard after = %f \n' %(jac_after))

        moving_nii = nib.Nifti1Image(imgs[0,:,:,:], np.eye(4))
        fixed_nii = nib.Nifti1Image(imgs[1,:,:,:], np.eye(4))
        warped_nii = nib.Nifti1Image(imgs[2,:,:,:], np.eye(4)) 

        #np.save (os.path.join(args.res_dir, 'label_moving_1_%i.npy' %i),label_moving_1)
        #np.save (os.path.join(args.res_dir, 'label_fixed_1_%i.npy' %(i)),label_fixed_1)
        #np.save (os.path.join(args.res_dir, 'label_warped_1_%i.npy' %(i)),warped_label_1)
        
    
        #nib.save(fixed_nii, os.path.join(args.res_dir, str(args.epochs) + '_fixed_uni_%i.nii' %(i)))                                                                                                      
        #nib.save(moving_nii, os.path.join(args.res_dir, str(args.epochs) + '_moving_uni_%i.nii' %(i)))                                                                                                    
        #nib.save(warped_nii, os.path.join(args.res_dir, str(args.epochs) + '_warped_uni_%i.nii' %(i)))                                                                                                    
        #np.save(os.path.join(args.res_dir, str(args.epochs) + '_deffield_uni_%i.npy' %(i)),imgs[3,:,:,:])                                                                                                 
        #np.save(os.path.join(args.res_dir, str(args.epochs) + '_flow_uni_%i.npy' %(i)),flow)

    print('DICE before = %f +- %f' %(np.mean(dices_before), np.std(dices_before)/np.sqrt(len(img_pairs))))
    print('DICE after = %f +- %f' %(np.mean(dices_after), np.std(dices_after)/np.sqrt(len(img_pairs))))

    print('Diff dice = %f' %(np.mean(dices_after)-np.mean(dices_before)))

    #print('JACCARD before = %f +- %f' %(np.mean(jacs_before), np.std(jacs_before)/np.sqrt(len(img_pairs))))
    #print('JACCARD after = %f +- %f' %(np.mean(jacs_after), np.std(jacs_after)/np.sqrt(len(img_pairs))))

    print('Number of pairs = %i' %(len(img_pairs)))

    """
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
    
   """
