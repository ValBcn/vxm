import os, sys
import argparse
import glob

sys.path.append('/home/valentin/CapsuleNetworkDetSeg/Segmentation')
sys.path.append('/home/valentin/local_packages/keras_med_io')

from capsnets_laseg.io.io_img_only import LocalPreprocessingBinarySeg


'''
 *
 * AUTHOR : Jordina Torrents-Barrena
 * E-MAIL : jordina.torrents@upf.edu
 * UNIVERSITY : Universitat Pompeu Fabra (UPF)
 * DATE : 17 / 12 / 2019
 *
'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="For preprocessing the dataset.")
    parser.add_argument("--dset_path", type = str, required = True, help = "Path to the base directory where you want to read your dataset.")
    parser.add_argument("--output_path", type = str, required = True, help = "Path to the base directory where you want to save your preprocessed dataset.")
    args = parser.parse_args()
    
    input_dir = os.path.join(args.dset_path, "original")
    output_dir = os.path.join(args.output_path, "preprocessed")
    print("Input Directory: ", input_dir, "\nOutput Directory: ", output_dir)
    
    preprocess = LocalPreprocessingBinarySeg(input_dir, output_dir)
    # removing the weird files that start with .__
    training_dir = os.path.join(input_dir, 'imagesTr')
    for filename in glob.glob(training_dir + "/._*"):
        os.remove(filename)
    # Preprocessing the dataset
    preprocess.gen_data()
