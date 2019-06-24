#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import numpy as np

from random import sample
from os import listdir, environ
from os.path import isfile, join
from skimage import img_as_float, img_as_ubyte
from network_model import inpainting_DLNet
from custom_ops import custom_psnr


# Collect all training, validation and test datafiles
def collect_datafiles(data_path):
    train_datafiles = [f for f in listdir(data_path) 
        if isfile(join(data_path, f)) and f[:len('train')]=='train']
    test_datafiles = [f for f in listdir(data_path) 
        if isfile(join(data_path, f)) and f[:len('test')]=='test']
    return sample(train_datafiles, len(train_datafiles)), test_datafiles


# Load data in specific .npz file
def load_data(data_path, filename, shuffle=False):
    npzfile = np.load(join(data_path, filename), 'r') 
    image_count = len(npzfile['images'])
    image_array = sample(npzfile['images'], image_count) \
        if shuffle else npzfile['images']
    return image_array
    

# Collect a minibatch of data 
def get_batch(images, batch_k, batch_size):
    return images[batch_k * batch_size : (batch_k + 1) * batch_size]


def main(args): 

    # environ['CUDA_VISIBLE_DEVICES']='' 
    assert (1000 % args.test_batch_size) == 0

    # Collect datafiles and setup test model
    _, test_filenames = collect_datafiles(args.data_path)
    test_model_options = {
        'batch_size' : args.test_batch_size,
        'image_shape' : [args.patch_size,]*2 + [3,],
        'phase' : 'Test',
        'reuse' : False,
        'k': args.k,
    }   
    # Input and output of the test model
    test_input_tensors, test_output_tensors = inpainting_DLNet(test_model_options)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:   

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if args.resume_path is None:
            print ('Resume path must be provided!')
        else:
            saver.restore(sess, join(args.resume_path, 'model_500.ckpt'))

        test_l1_loss_cum, test_l2_loss_cum, test_psnr_cum = 0, 0, 0
        test_batch_no = 0             
        test_images = load_data(args.data_path, test_filenames[0], shuffle=False)
        while (test_batch_no+1)*args.test_batch_size <= len(test_images):
            # Collect input image array
            image_array = get_batch(test_images, test_batch_no, args.test_batch_size)
            # Collect generated image array
            fake_image_array = sess.run(test_output_tensors['fake_images'], 
            feed_dict = {
                test_input_tensors['images'] : img_as_float(image_array),
                test_input_tensors['inpaint_size_range'] : [20, 20],
                test_input_tensors['inpaint_shift_range'] : [0, 0],
            })
            test_batch_no += 1
            # Images are scaled to [-1, +1] so x2
            test_l1_loss_cum += np.mean(np.absolute(fake_image_array - img_as_float(image_array))*2)
            test_l2_loss_cum += np.mean(np.square(fake_image_array - img_as_float(image_array))*4)                    
            fake_image_array = np.maximum(np.minimum(fake_image_array, 1.0), 0.0)
            fake_image_array = img_as_ubyte(fake_image_array)
            test_psnr_cum += np.sum(custom_psnr(fake_image_array, image_array))
        print ('Test l1: %s, RMSE: %s, PSNR: %s.' %(
               str(test_l1_loss_cum/test_batch_no), 
               str(test_l2_loss_cum/test_batch_no), 
               str(test_psnr_cum/test_batch_no/args.test_batch_size)))  

    tf.reset_default_graph()
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--k', type=int,
        help='Number of gradient descent steps for updating z.', default=5)    
    parser.add_argument('--max_epoch', type=int,
        help='Number of training epochs.', default=500)
    parser.add_argument('--snapshot_interval', type=int, 
        help='Snapshot per X epochs during training.', default=100)
    parser.add_argument('--decay_interval', type=int,
        help='Decay per X epochs during training.', default=200)
    parser.add_argument('--test_interval', type=int, 
        help='Test per X epochs during training.', default=10)
    parser.add_argument('--log_interval', type=int, 
        help='Log per X epochs during training.', default=1)
    parser.add_argument('--patch_size', type=int,
        help='Patch height/width in pixels.', default=64)
    parser.add_argument('--batch_size', type=int,
        help='Number of samples per minibatch for training.', default=100)
    parser.add_argument('--test_batch_size', type=int, 
        help='Number of samples per minibatch for test.', default=50)
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate', default=1e-3)
    parser.add_argument('--decay_rate', type=float,
        help='Learning rate decay rate', default=1e-1)
    parser.add_argument('--momentum', type=float, 
        help='Solver momentum', default=0.9)
    parser.add_argument('--from_scratch', 
        help='Let deep likelihood net being trained from scratch.', action='store_true')

    parser.add_argument('--resume_path', type=str,
        help='Directory for your pretrained model.', default='./ckts/')  
    parser.add_argument('--data_path', type=str,
        help='Directory for training and test data.', default='../Data/CelebA/numpy/')
    parser.add_argument('--ckt_dir', type=str,
        help='Directory for ckt files.', default='ckts')

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
