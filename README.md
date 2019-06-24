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
Some important arguments for our training scripts are:
* `--k`: Number of gradient descent iterations for updating z, i.e., k.
* `--model_path`: Directory of your pre-trained model.
* `--data_path`: Directory of your npz data.
* `--from_scratch`: Whether to train from scatch (default false).
* `--learning_rate`: Initial learning rate (optional).
* `--max_epoch`: Number of training epochs (optional).

An example of training image inpainting autoencoder using our DL-Net:
```
python train_inpaint.py \
--model_path ./path/to/your/pretrained/ckts/ \
--data_path ./path/to/your/data/ \
--k 5
```
With k=1, it boils down to the ridgt-joint training as explained:
```
python train_inpaint.py \
--model_path ./path/to/your/pretrained/ckts/ \
--data_path ./path/to/your/data/ \
--k 1
```
Pretrained models used in our experiments can be downloaded at:

[CelebA pretrained inpainting model](https://drive.google.com/open?id=1Udu4dB_YFF2MscfrcbfWeQ1HxoMY7HUs) \
[CelebA pretrained interpolation model](https://drive.google.com/open?id=1UwBxo7tdxIUfNhmf22iihzEoyTmsHX8X) \
[SUN397 pretrained inpainting model](https://drive.google.com/open?id=1xzMgAkhSNCvYXSdKbWLDAlzHIiAe-bxP) \
[SUN397 pretrained interpolation model](https://drive.google.com/open?id=1yWt-zyUS3uSMrGGaWl82dzXxsG_4BGyO)

These models are trained to process images under one specific degradation level. You can put them in any directory you prefer and pass its path to the argument `--model_path` of `train_*.py`. Notably, our DL-Net can also be trained from scratch, for examply by simply running:
```
python train_inpaint.py \
--data_path ./path/to/your/data/ \
--from_scratch \
--k 5
```
Test is performed by running another python script. An example of testing our DL-Net flavoured image inpainting autoencoder:
```
python test_inpaint.py \
--resume_path ./path/to/your/DL-NET/ckts/ \
--k 5
```
Scripts for training and testing image interpolation DL-Net models are also provided. For SISR, the core [block of code](https://github.com/yiwenguo/Deep-Likelihood-Network/blob/74561ce6d667107ef822d61280751924233231db/network_model.py#L211) is similar. If you have any question regarding this problem, contact me at yiwen.guo@intel.com.

# Citation
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
