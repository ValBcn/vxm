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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model
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
    parser.add_argument("--loss_dir", type = str, required = True,
                        help = "Path to the directory where the loss is saved")
    parser.add_argument("--batch_size", type = int, required = False, default = 2,
                        help = "Batch size for the CNN should be 17 and for the Capsule Network, it should be 2.")
    parser.add_argument("--lr", type = float, required = False, default = 1e-3,
                        help = "The learning rate")
    parser.add_argument("--trainset", type = float, required = False, default = .6,
                        help = "Size of the training set.")
    parser.add_argument("--valset", type = float, required = False, default = .2,
                        help = "Size of the validation set.")
    parser.add_argument("--testset", type = float, required = False, default = .2,
                        help = "Size of the testing set.")
    parser.add_argument("--multi", type = boolean_string, required = False, default = False,
                        help = "multi or uni modal registration.")
    parser.add_argument("--in_weights", type = str, required = False,
                        help = " path to pretrained weights")
    parser.add_argument("--in_model", type = str, required = False,
                        help = " path to pretrained weights")
    parser.add_argument("--lambda_", type = float, required = False, default = 0.01,
                        help = " weight of the gradient normalization of the registration field")


    args = parser.parse_args()

    # loading data

    img_path1 = os.path.join(args.img_path1)
    if args.multi is True:
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
    losses = [vxm.losses.MutualInformation().loss, vxm.losses.Grad('l2').loss]
    loss_weights = [1, args.lambda_]

    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
        return lr

    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0, nesterov=False, name="SGD")
    #optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr,rho=0.9,momentum=0.9,epsilon=1e-07,centered=False,name="RMSprop")
    #optimizer = tf.keras.optimizers.Adadelta(learning_rate=args.lr, rho=0.95, epsilon=1e-07, name="Adadelta")



    lr_metric = get_lr_metric(optimizer)
    
    if args.in_model is None:
        vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy',lr_metric])
    else:
        vxm_model.load_weights(args.in_model)
        vxm_model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=['accuracy',lr_metric])
        
    if args.multi == True:
        train_generator = vxm_data_generator(x_train,x_train2, batch_size=2)
    if args.multi == False:
        train_generator = vxm_data_generator(x_train,x_train, batch_size=2)

    in_sample, out_sample = next(train_generator)

    print('input shape: ', ', '.join([str(t.shape) for t in vxm_model.inputs]))
    print('output shape:', ', '.join([str(t.shape) for t in vxm_model.outputs]))
    
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.9, patience=10, min_lr=1e-7, verbose=1)
    EarlyStopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(os.path.join(args.weights_dir, "checkpoint_"+ str(args.epochs))),
                                                                   save_weights_only=True,
                                                                   monitor='loss',
                                                                   mode='min',
                                                                   save_best_only=True,
                                                                   verbose=1)
    
    hist = vxm_model.fit(train_generator, epochs=args.epochs, steps_per_epoch=5, verbose=2, callbacks=[reduce_lr,model_checkpoint_callback])
    loss = hist.history["loss"]
    loss_np = np.array(loss)
    lr = hist.history["lr"]
    lr_np = np.array(lr)
    
    if args.multi == True:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5"))
        np.savetxt(str(os.path.join(args.loss_dir, "loss_" + str(args.epochs) + "epochs_multi.txt")),loss_np)
        np.savetxt(str(os.path.join(args.loss_dir, "lr_" + str(args.epochs) + "epochs_multi.txt")),lr_np)
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5")))
    if args.multi == False:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5"))
        np.savetxt(str(os.path.join(args.loss_dir, "loss_" + str(args.epochs) + "epochs_uni.txt")),loss_np)
        np.savetxt(str(os.path.join(args.loss_dir, "lr_" + str(args.epochs) + "epochs_uni.txt")),lr_np)
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5")))

    

    
"""
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
from scripts.vxm_gen import *

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="For training on Fetal MRI Segmentation using Capsule Networks and CNNs")
    parser.add_argument("--weights_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save your weights")
    parser.add_argument("--loss_dir", type = str, required = True,
                        help = "Path to the base directory where you want to save loss history")
    parser.add_argument("--img_path1", type = str, required = True,
                        help = "Path to the base directory where the preprocessed/reshaped images are")
    parser.add_argument("--label_path1", type = str, required = False,
                        help = "Path to the base directory where the preprocessed/reshaped labels are")
    parser.add_argument("--img_path2", type = str, required = False,
                        help = "If bi-model registration. Path to the base directory where the preprocessed/reshaped images 2 are")
    parser.add_argument("--label_path2", type = str, required = False,
                        help = "Path to the base directory where the preprocessed/reshaped labels are")
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
    parser.add_argument("--with_labels", type = boolean_string, required = False, default = False,
                        help = "add labels (semi-supervised learning)")
    
    args = parser.parse_args()


    # loading data
    
    img_path1 = os.path.join(args.img_path1)

    if args.multi is True:
        img_path2 = os.path.join(args.img_path2)
        if args.with_labels is True:
            data = [[],[]] # lists with images + labels for modality 1
            data2 = [[],[]] # list with images + labels for modality 2
        if args.with_labels is False:
            data = [] # lists with images for modality 1                                                                                                                                        
            data2 = [] # list with images + labels for modality 2
    if args.multi is False:
        if args.with_labels is True:
            data = [[],[]]
        if args.with_labels is False:
            data = []

    for img in os.listdir(img_path1):
        img_array = np.load(os.path.join(img_path1,img))
        if args.with_labels is True:
            data[0].append(img_array)
        if args.with_labels is False:
            data.append(img_array)
    if args.with_labels is True:
        label_path1 = os.path.join(args.label_path1)
        for lab in os.listdir(label_path1):
            label_array = np.load(os.path.join(label_path1,lab))
            data[1].append(label_array)
            
    data = np.stack(data)
    data = np.asarray(data)

    if args.with_labels is True:
        data = np.moveaxis(data,0,-1)
        
    print ('data shape =', data.shape)

    x_train, x_val = train_test_split(data, test_size=0.2)    

    if args.with_labels is True:
        vol_shape = x_train.shape[1:4]
    if args.with_labels is False:
        #vol_shape = x_train.shape[1:]
        vol_shape = img_array['vol'].shape
    print('train shape', x_train.shape)
    print('val shape', x_val.shape)
    print('vol_shape',vol_shape)

    if args.multi is True:
        for img in os.listdir(img_path2):
            img_array2 = np.load(os.path.join(img_path2,img))
            if args.with_labels is True:
                data2[0].append(img_array2)
            if args.with_labels is False:
                data2.append(img_array2)
        if args.with_labels is True:
            label_path2 = os.path.join(args.label_path2)
            for lab in os.listdir(label_path2):
                label_array2 = np.load(os.path.join(label_path2,lab))
                data2[1].append(label_array2)
            
        #data2[0] = np.stack(data2[0])
        #data2[1] = np.stack(data2[1])
        data2 = np.asarray(data2)

        x_train2, x_val2 = train_test_split(data2, test_size=0.2)

        if args.with_labels is True:
            vol_shape2 = x_train2.shape[2:]
        if args.with_labels is False:
            vol_shape2 = x_train2.shape[1:]

        print('train_2 shape', x_train2.shape)
        print('val_2 shape', x_val2.shape)
        print('vol_shape_2',vol_shape2)


    # configure unet input shape (concatenation of moving and fixed images)
    ndim = 3
    unet_input_features = 4
    inshape = (*x_train.shape[1:], unet_input_features)

    # configure unet features 
    nb_features = [
        [32, 32, 32, 32],         # encoder features
        [32, 32, 32, 32, 32, 16]  # decoder features
    ]

    if args.with_labels is False:
        if args.multi == True:
            train_generator = vxm_data_generator(x_train,x_train2, batch_size=2)
        if args.multi == False:
            train_generator = vxm_data_generator2(x_train, batch_size=2)
        in_sample, out_sample = next(train_generator)
    if args.with_labels is True:
        if args.multi == True:
            train_generator = vxm_data_generator_w_labels(x_train,x_train2, batch_size=2)
        if args.multi == False:
            train_generator = vxm_data_generator_old(x_train, batch_size=2)
        #in_sample, out_sample = next(train_generator)

    
    #print('in sample shape =', np.asarray(in_sample).shape)
    #print('out sample shape =', np.asarray(out_sample).shape)
    
    # unet
    if args.with_labels is True:
        vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(vol_shape)
    if args.with_labels is False:
        vxm_model = vxm.networks.VxmDenseSemiSupervisedSeg(inshape=vol_shape, nb_unet_features=nb_features, int_steps=0, nb_labels=1)

    # losses and loss weights
    losses = [vxm.losses.NCC().loss,vxm.losses.NCC().loss,vxm.losses.NCC().loss]
    loss_weights = [0.01, 0.01, 0.01]

    vxm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=args.lr), loss=losses, loss_weights=loss_weights)
    
    hist = vxm_model.fit(train_generator, epochs=args.epochs, steps_per_epoch=5, verbose=2)

    loss = hist.history["loss"]

    loss_np = np.array(loss)
    
    if args.multi == True:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5"))
        np.savetxt(str(os.path.join(args.loss_dir, "loss_" + str(args.epochs) + "epochs_multi.txt")),loss_np)
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_multi.h5")))
    if args.multi == False:
        vxm_model.save_weights(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5"))
        np.savetxt(str(os.path.join(args.loss_dir, "loss_" + str(args.epochs) + "epochs_uni.txt")),loss_np)
        print ('saved weights as', str(os.path.join(args.weights_dir, str(args.epochs) + "epochs_uni.h5")))
    
    
"""
