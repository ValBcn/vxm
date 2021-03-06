import numpy as np
import nibabel as nib
import itertools
import os
import argparse

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def resize_data(data,new_size_x,new_size_y,new_size_z):
    initial_size_x = data.shape[0]
    initial_size_y = data.shape[1]
    initial_size_z = data.shape[2]

    delta_x = initial_size_x / new_size_x
    delta_y = initial_size_y / new_size_y
    delta_z = initial_size_z / new_size_z

    new_data = np.zeros((new_size_x, new_size_y, new_size_z))

    for x, y, z in itertools.product(range(new_size_x),
                                     range(new_size_y),
                                     range(new_size_z)):
        new_data[x][y][z] = data[int(x * delta_x)][int(y * delta_y)][int(z * delta_z)]

    return new_data


    img_path = os.path.join(args.dset_path)
    img_path_us = os.path.join(args.dset_path_us)
    data = []
    data_us = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reshaping the data to fit the network (and reduce computational cost if needed)")
    parser.add_argument("--img_path1", type = str, required = True,
                        help = "Path to the base directory where the preprocessed input are.")
    parser.add_argument("--img_path2", type = str, required = False,
                        help = "For multi-modal registration, path to the base directory where the second preprocessed inputs are.")
    parser.add_argument("--multi", type = boolean_string, required = False, default = False,
                        help = "multi or uni modal registration.")
    parser.add_argument("--reshape", type = int, required = False, default = 64,
                        help = "size x of the reshaping (new shape = (x/2,x,x), x musty me a multiple of 16.")
    parser.add_argument("--reshaped_path1", type = str, required = True,
                        help = "Path to the directory where the reshaped imgs 1 are.")
    parser.add_argument("--reshaped_path2", type = str, required = False,
			help = "Path to the directory where the reshaped imgs 2 are.")



    args = parser.parse_args()


    # Removing older files
    os.system('rm %sreshaped_*' %(args.img_path1))
    
    # Resizing MRI                                                                                                                                                                                                
    print ('Reshaping the inputs 1')
    i = 1
    len_dir1 = len(os.listdir(args.img_path1))
    for img in os.listdir(args.img_path1):
        img_array = np.load(os.path.join(args.img_path1,img), allow_pickle = True)
        if len(img_array.shape) < 3:
            img_array = np.load(os.path.join(args.img_path1,img), allow_pickle = True)[0]
        print ('%i/%i' %(i, len_dir1))
        img_array = resize_data(img_array,int(args.reshape/2),args.reshape,args.reshape)
        np.save(os.path.join(args.reshaped_path1, 'reshaped_' + str(args.reshape) + img),img_array)
        i += 1
        
    if args.multi is True:
        os.system('rm %sreshaped_*' %(args.img_path2))
    # Resizing US
        print ('Reshaping the inputs 2')
        i = 1
        len_dir2 = len(os.listdir(args.img_path2))
        for img in os.listdir(args.img_path2):
            img_array2 = np.load(os.path.join(args.img_path2,img), allow_pickle = True)
            if len(img_array2.shape) < 3:
                img_array2 = np.load(os.path.join(args.img_path2,img), allow_pickle = True)[0]
            print ('%i/%i' %(i, len_dir2))
            img_array2 = resize_data(img_array2,int(args.reshape/2),args.reshape,args.reshape)
            np.save(os.path.join(args.reshaped_path2, 'reshaped_' + str(args.reshape) + img),img_array2)
            i += 1
