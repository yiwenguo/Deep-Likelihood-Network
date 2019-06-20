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
    if args.model_path is not None and args.from_scratch:
        print ('Sure you want to train from scratch?')

    # Collect datafiles and setup training model
    filenames, test_filenames = collect_datafiles(args.data_path)
    model_options = {
        'batch_size' : args.batch_size,
        'image_shape' : [args.patch_size,]*2 + [3,],
        'phase' : 'Train',
        'reuse' : False,
        'k': args.k,
    }    
    # Input and output of the training model
    input_tensors, output_tensors = inpainting_DLNet(model_options)

    # Setup learning rate
    global_step = tf.Variable(0, trainable=False)     
    global_step_op = tf.assign(global_step, global_step + 1)  
    learning_rate = tf.train.exponential_decay(
        args.learning_rate, 
        global_step=global_step,
        decay_steps=args.decay_interval, 
        decay_rate=args.decay_rate, 
        staircase=True
    )        

    # Setup optimizer
    all_vars = [var for var in tf.trainable_variables() 
        if var.name.startswith('net_')]        
    all_loss = output_tensors['eucd_loss'] + output_tensors['norm_loss']
    optimizer = tf.train.AdamOptimizer(learning_rate)    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(all_loss, var_list=all_vars) 
        
    # # Collect test data and setup test model
    # assert len(test_filenames)==1
    # test_images = load_data(args.data_path, test_filenames[0], shuffle=False)
    # test_model_options = {
    #     'batch_size' : args.test_batch_size,
    #     'image_shape' : [args.patch_size,]*2 + [3,],
    #     'phase' : 'Test',
    #     'reuse' : True,
    #     'k': args.k,
    # }    
    # # Input and output of the test model
    # test_input_tensors, test_output_tensors = inpainting_DLNet(test_model_options) 

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:   
        sess.run(tf.global_variables_initializer())

        # Load pre-trained model for a single degradation setting
        if args.from_scratch is False:
            pretrained_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net_')
            pre_saver = tf.train.Saver(pretrained_vars)
            pre_saver.restore(sess, join(args.model_path, 'model.ckpt'))          

        saver = tf.train.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) 

        for i in xrange(args.max_epoch):            
            batch_no_cum, eucd_loss_cum = 0, 0
            for j in xrange(len(filenames)):
                batch_no = 0                
                images = load_data(args.data_path, filenames[j], shuffle=True)
                while (batch_no+1)*args.batch_size <= len(images):
                    image_array = get_batch(images, batch_no, args.batch_size)
                    eucd_loss = sess.run([train_op, output_tensors['eucd_loss']], 
                    feed_dict = {
                        input_tensors['images'] : img_as_float(image_array),
                        input_tensors['inpaint_size_range'] : [1, 30],
                        input_tensors['inpaint_shift_range'] : [-10, 10],
                    })[1]
                    batch_no += 1
                    eucd_loss_cum += eucd_loss
                batch_no_cum += batch_no
            filenames = sample(filenames, len(filenames))

            if (i+1) % args.log_interval == 0:
                print ('Epoch: %d; Training Euclidean loss: %s.' 
                       %(i+1, str(eucd_loss_cum/batch_no_cum)))
            
            if (i+1) % args.snapshot_interval == 0:
                print ("Saving model .............")
                saver.save(sess, args.ckt_dir+'/model_%d.ckpt' %(i+1))

            # # Test is performed in test_context_encoder.py
            # if (i+1) % args.test_interval == 0:
            #     test_batch_no, test_eucd_loss_cum, test_psnr_cum = 0, 0, 0
            #     while (test_batch_no+1)*args.test_batch_size <= len(test_images):
            #         test_image_array = get_batch(test_images, test_batch_no, args.test_batch_size)
            #         fake_image_array, test_eucd_loss  = \
            #         sess.run([test_output_tensors['fake_images'],
            #                   test_output_tensors['eucd_loss']], 
            #         feed_dict = {
            #             test_input_tensors['images'] : img_as_float(test_image_array),
            #             test_input_tensors['inpaint_size_range'] : [20, 20],
            #             test_input_tensors['inpaint_shift_range'] : [0, 0],
            #         })                    
            #         test_batch_no += 1
            #         test_eucd_loss_cum += test_eucd_loss
            #         fake_image_array = np.maximum(np.minimum(fake_image_array, 1.0), 0.0)
            #         fake_image_array = img_as_ubyte(fake_image_array)
            #         test_psnr_cum += np.sum(custom_psnr(fake_image_array, test_image_array))
            #     print ('Test Euclidean loss: %s, PSNR: %s.' %(str(test_eucd_loss_cum/test_batch_no), 
            #            str(test_psnr_cum/test_batch_no/args.test_batch_size)))

            sess.run(global_step_op)
            sys.stdout.flush()            

        coord.request_stop()
        coord.join(threads)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('k', type=int,
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

    parser.add_argument('--model_path', type=str,
        help='Directory for your pretrained model.', default=None)  #'./pre_trained/ckts') 
    parser.add_argument('--data_path', type=str,
        help='Directory for training and test data.', default='./data/CelebA/numpy/')
    parser.add_argument('--ckt_dir', type=str,
        help='Directory for ckt files.', default='ckts')

    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))
