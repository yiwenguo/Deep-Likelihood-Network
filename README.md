# Deep-Likelihood-Network
This is a TensorFlow implementation of deep likelihood network for image restoration with multiple restorations.

# Environment Prerequisite
* CUDA 9.0 & cuDNN v7
* Tensorflow 1.7.0
* Python 2.7
* scikit-image

# Prepare Data
We use [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [SUN397](https://groups.csail.mit.edu/vision/SUN/) in the image inpainting and interpolation experiments. For CelebA, the aligned images are used and the first 100,000 images sorted automatically by `os.listdir(path)` is used for training. If the training, validation, and test images are stored at `./data/CelebA/train/images`, `./data/CelebA/val/images`, and `./data/CelebA/test/images`, respectively, then simply run the python script to generate .npz files:
```
python create_celeba_npz.py
``` 
Files for training and test with SUN397 can be generated similarly, and the path to the .npz files should be passed to the argument `--data_path` of `train_*.py`.

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
--model_path ./path/to/your/pretrained/model/ \
--data_path ./path/to/your/data/ \
--k 5
```
With k=1, it boils down to the ridgt-joint training as explained:
```
python train_inpaint.py \
--model_path ./path/to/your/pretrained/model/ \
--data_path ./path/to/your/data/ \
--k 1
```
Pretrained models used in our experiments can be downloaded at:

[CelebA pretrained inpainting model](https://drive.google.com/open?id=1Udu4dB_YFF2MscfrcbfWeQ1HxoMY7HUs) \
[CelebA pretrained interpolation model](https://drive.google.com/open?id=1rqvW3EAhocIpxLRfJQON4XFU89mWf3S0) \
[SUN397 pretrained inpainting model](https://drive.google.com/open?id=1LkWpkQMmL4UW21ztpXncZo_nvnfVl2VD) \
[SUN397 pretrained interpolation model](https://drive.google.com/open?id=1Ly3a_OhmnOaqXP3rXmD5XhC9VCBqzEdq)

These models are trained to process images under one specific degradation level. You can put them in any directory you perfer and pass its path to the argument `--model_path` of `train_*.py`.

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
