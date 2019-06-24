import tensorflow as tf
import numpy as np


# Calauclate PSNR
def custom_psnr(x, x_hat):
    #assume RGB image
    diff = (np.array(x).astype(np.float64) - np.array(x_hat).astype(np.float64))
    diff = np.reshape(diff, [diff.shape[0], -1])
    rmse = np.sqrt(np.mean(diff**2., 1))
    return 20*np.log10(255.0/rmse)


# Output Gaussian blurred images
def custom_gaussian_blur(x, width, sigma):
    # For now, only kernels with odd width are supported
    half_width = (width-1)/2
    batch_size = x.get_shape().as_list()[0]
    sigma = tf.reshape(sigma, [1,1,batch_size])
    # Pad the input images
    x_padded = tf.pad(x, [[0,0],[half_width,half_width], 
               [half_width,half_width],[0,0]], 
               mode='SYMMETRIC')

    # Get the kernel matrices
    s = tf.constant(range(width), dtype=tf.float32) - (width-1)/2
    sx = tf.tile(tf.expand_dims(s, 1), [1, width])
    sy = tf.tile(tf.expand_dims(s, 0), [width, 1])
    kernels = tf.exp(tf.expand_dims(-(sx*sx + sy*sy), 2)/(2*sigma*sigma))
    kernels = kernels/tf.reduce_sum(kernels, [0,1], keep_dims=True)

    images_blurred = []
    for i in xrange(batch_size):
        image = tf.expand_dims(x_padded[i,:,:,:], 0)
        kernel = tf.reshape(kernel[:,:,i], [width,width, 1, 1])
        images_blurred.append(tf.nn.conv2d(image, kernel, 
            strides=[1, 1, 1, 1], padding='VALID'))
    y = tf.reshape(tf.stack(images_blurred), x.get_shape().as_list())

    return y