# Deep-Likelihood-Network
This is a TensorFlow implementation of deep likelihood network for image restoration with multiple restorations.

# Environment Prerequisite
* CUDA 9.0 & cuDNN v7
* Tensorflow 1.7.0
* Python 2.7
* scikit-image

# Prepare Data
We use [[CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)] and [[SUN397](https://groups.csail.mit.edu/vision/SUN/)] in the image inpainting and interpolation experiments. For CelebA, the aligned images are used and the first 100,000 images sorted automatically by `os.listdir(path)` is used for training. If your training, validation and test images are stored at `./data/CelebA/train/images`, `./data/CelebA/val/images`, and `./data/CelebA/test/images`, respectively, then run the provided python script to generate .npz files at `./data/CelebA/numpy':
```
python create_celeba_npz.py
``` 

# Run experiments
Some important arguments for our main script are:
* `--k`: Number of gradient descent iterations for updating z, i.e., k.
* `--model_path`: Directory of your pre-trained model.
* `--data_path`: Directory of your npz data.
* `--learning_rate`: Initial learning rate (optional).
* `--max_epoch`: Number of training epochs (optional).

An example of training image inpainting autoencoder using our DL-Net:
```
python train_inpaint.py \
./path/to/your/pretrained/model/ \
./path/to/your/data/ \
--k 5
```
With k=1, it boils down to the ridgt-joint training as explained:
```
python train_inpaint.py \
./path/to/your/pretrained/model/ \
./path/to/your/data/ \
--k 1
```
Pretrained models used in our experiments can be downloaded at:

# Citation and contact
Please cite our work in your publications if it helps your research:
```
@article{guo2019deep,
  title={Deep Likelihood Network for Image Restoration with Multiple Degradations},
  author={Guo, Yiwen and Zuo, Wangmeng and Zhang, Changshui and Chen, Yurong},
  journal={arXiv preprint arXiv:1904.09105},
  year={2019}
}
```

## License
MIT License
