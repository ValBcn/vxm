# unimodal registration (semi-supervised)

#python main.py --epochs 2000 --img_path1 data_npz/  --weights_dir weights/ --loss_dir loss/ --multi False


# bimodal registration

#python main.py --epochs 200 --img_path1 reshaped_data/mri_img/ --img_path2 reshaped_data/us_img/ --weights_dir weights/ --loss_dir loss/ --multi True


# unimodal registration

python main.py --epochs 500 --img_path1 $1 --weights_dir $2 --loss_dir $3 --multi False --batch_size 16 --lr 1e-3 --lambda 1e-2 #--in_model $4
