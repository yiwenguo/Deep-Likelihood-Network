import tensorflow as tf


def generate_inpainting_mask(batch_size, image_shape, inpaint_size_range, inpaint_shift_range):    
    random_inpaint_size = tf.random_uniform([batch_size,], 
                                            minval=inpaint_size_range[0], 
                                            maxval=inpaint_size_range[1] + 1.0) 
    random_inpaint_size = tf.cast(tf.floor(random_inpaint_size), dtype=tf.int32)     
    random_inpaint_shift = tf.random_uniform([batch_size,2], 
                                             minval=inpaint_shift_range[0], 
                                             maxval=inpaint_shift_range[1] + 1.0) 
    random_inpaint_shift = tf.cast(tf.floor(random_inpaint_shift), dtype=tf.int32) 

    mask_height = mask_width = random_inpaint_size    
    h = tf.round((image_shape[0] - mask_height)/2 + random_inpaint_shift[:,0]) 
    v = tf.round((image_shape[1] - mask_width)/2 + random_inpaint_shift[:,1]) 
    hmask = tf.cast(tf.expand_dims(tf.range(image_shape[0]), 0) >= 
                    tf.expand_dims(h, 1), dtype=tf.float32) * \
            tf.cast(tf.expand_dims(tf.range(image_shape[0]), 0) < 
                    tf.expand_dims(h + mask_height, 1), dtype=tf.float32)
    vmask = tf.cast(tf.expand_dims(tf.range(image_shape[1]), 0) >= 
                    tf.expand_dims(v, 1), dtype=tf.float32) * \
            tf.cast(tf.expand_dims(tf.range(image_shape[1]), 0) < 
                    tf.expand_dims(v + mask_width, 1), dtype=tf.float32)
    image_mask = tf.expand_dims(hmask, 2) * tf.expand_dims(vmask, 1)
    image_mask = tf.reshape(image_mask, [batch_size,] + image_shape[:2] + [1,])

    return image_mask


def generate_interpolation_mask(batch_size, image_shape, interpo_perc_range):    
    random_inpaint_perc = tf.random_uniform([batch_size,1,1,1], 
                                            minval=interpo_perc_range[0], 
                                            maxval=interpo_perc_range[1]) 

    image_mask = tf.random_uniform([batch_size,]+image_shape[:2]+[1,], 
                                   minval=0.0, maxval=1.0)
    image_mask = tf.cast(image_mask <= random_inpaint_perc, dtype=tf.float32)

    return image_mask


def generate_weights(feature_dims):
    Weights = []
    for i in xrange(4):  
        Weights.append(tf.get_variable('conv%s_enc' %i + '/W', 
            shape=[4, 4, feature_dims[i], feature_dims[i+1]],
            initializer=tf.contrib.layers.variance_scaling_initializer(), 
            dtype=tf.float32))
        Weights.append(tf.get_variable('conv%s_enc' %i + '/b', 
            shape=[feature_dims[i+1],],
            initializer=tf.constant_initializer(0.0), 
            dtype=tf.float32))
    Weights.append(tf.get_variable('conv_fc' + '/W', 
        shape=[feature_dims[-1], 16, 16],
        initializer=tf.random_normal_initializer(stddev=0.001), 
        dtype=tf.float32))
    Weights.append(tf.get_variable('conv_fc' + '/b', 
        shape=[feature_dims[-1],],
        initializer=tf.constant_initializer(0.0), dtype=tf.float32))
    for i in xrange(4):  
        Weights.append(tf.get_variable('conv%s_dec' %i + '/W', 
            shape=[4, 4, feature_dims[-(i+2)], feature_dims[-(i+1)]],
            initializer=tf.contrib.layers.variance_scaling_initializer(), 
            dtype=tf.float32))
        Weights.append(tf.get_variable('conv%s_dec' %i + '/b', 
            shape=[feature_dims[-(i+2)],],
            initializer=tf.constant_initializer(0.0), 
            dtype=tf.float32))

    return Weights


def inpainting_net(Inputs, Weights, first_iter=False, previous_z_dec=None, is_train=False):
    
    input_images, inpaint_size, inpaint_shift, image_mask = Inputs  
    masked_images = input_images * (1.0 - image_mask)
    input_shape = input_images.get_shape().as_list()
    batch_size = input_shape[0]

    # The context encoder
    if first_iter:
        z = masked_images
        for i in xrange(4):                          
            z_enc = tf.nn.conv2d(z, Weights[2*i], strides=[1, 2, 2, 1], padding='SAME')
            z_enc = tf.layers.batch_normalization(z_enc, 
                training=is_train, name='bn%s_enc' %i)
            z = tf.nn.leaky_relu(z_enc, 0.2)   
        z = tf.transpose(tf.reshape(z, [batch_size, -1, 512]), [2, 0, 1])
        z = tf.reshape(tf.matmul(z, Weights[2*4]), [512, batch_size, 4, 4])
        z = tf.transpose(z, [1, 2, 3, 0]) 
        for i in xrange(3):              
            shape = [batch_size,8*(2**i),8*(2**i),Weights[2*i+10].get_shape().as_list()[-2]]
            z_dec = tf.nn.conv2d_transpose(z, Weights[2*i+10], 
                output_shape=shape, strides=[1, 2, 2, 1], padding='SAME') 
            z_dec = tf.layers.batch_normalization(z_dec, 
                training=is_train, name='bn%s_dec' %i)
            z = tf.nn.relu(z_dec)    
    else:
        assert previous_z_dec is not None
        z_dec = previous_z_dec
        z = tf.nn.relu(z_dec)

    # The estimation
    fake_images = tf.nn.conv2d_transpose(z, Weights[-2], 
        output_shape=input_shape, strides=[1, 2, 2, 1], padding='SAME')        

    eucd_loss = tf.reduce_sum(tf.square(fake_images - input_images))/batch_size/2 
    like_loss = tf.reduce_sum(tf.square(fake_images - input_images)*(1.0 - image_mask)  \
        /tf.reduce_sum(1.0 - image_mask, [1, 2, 3], keepdims=True))/batch_size/2
    conv_vars = [var for var in tf.trainable_variables() if '/conv' in var.name]              
    norm_loss = tf.reduce_sum(1e-4 * tf.stack([tf.nn.l2_loss(i) for i in conv_vars]))    

    input_tensors = {
        'images' : input_images,
        'inpaint_size_range' : inpaint_size,
        'inpaint_shift_range' : inpaint_shift,
    }

    output_tensors = {
        'fake_images' : fake_images,
        'fake_hidden' : z_dec,
        'like_loss' : like_loss,
        'eucd_loss' : eucd_loss,
        'norm_loss' : norm_loss,
    }

    return input_tensors, output_tensors


def interpolation_net(Inputs, Weights, first_iter=False, previous_z_dec=None, is_train=False):
    
    input_images, perc_range, image_mask = Inputs  
    masked_images = input_images * (1.0 - image_mask)
    input_shape = input_images.get_shape().as_list()
    batch_size = input_shape[0]

    # The context encoder
    if first_iter:
        z = masked_images
        for i in xrange(4):                          
            z_enc = tf.nn.conv2d(z, Weights[2*i], strides=[1, 2, 2, 1], padding='SAME')
            z_enc = tf.layers.batch_normalization(z_enc, 
                training=is_train, name='bn%s_enc' %i)
            z = tf.nn.leaky_relu(z_enc, 0.2)   
        z = tf.transpose(tf.reshape(z, [batch_size, -1, 512]), [2, 0, 1])
        z = tf.reshape(tf.matmul(z, Weights[2*4]), [512, batch_size, 4, 4])
        z = tf.transpose(z, [1, 2, 3, 0]) 
        for i in xrange(3):              
            shape = [batch_size,8*(2**i),8*(2**i),Weights[2*i+10].get_shape().as_list()[-2]]
            z_dec = tf.nn.conv2d_transpose(z, Weights[2*i+10], 
                output_shape=shape, strides=[1, 2, 2, 1], padding='SAME') 
            z_dec = tf.layers.batch_normalization(z_dec, 
                training=is_train, name='bn%s_dec' %i)
            z = tf.nn.relu(z_dec)    
    else:
        assert previous_z_dec is not None
        z_dec = previous_z_dec
        z = tf.nn.relu(z_dec)

    # The estimation
    fake_images = tf.nn.conv2d_transpose(z, Weights[-2], 
        output_shape=input_shape, strides=[1, 2, 2, 1], padding='SAME')        

    eucd_loss = tf.reduce_sum(tf.square(fake_images - input_images))/batch_size/2 
    like_loss = tf.reduce_sum(tf.square(fake_images - input_images)*(1.0 - image_mask)  \
        /tf.reduce_sum(1.0 - image_mask, [1, 2, 3], keepdims=True))/batch_size/2
    conv_vars = [var for var in tf.trainable_variables() if '/conv' in var.name]              
    norm_loss = tf.reduce_sum(1e-4 * tf.stack([tf.nn.l2_loss(i) for i in conv_vars]))    

    input_tensors = {
        'images' : input_images,
        'interpo_perc_range' : perc_range,
    }

    output_tensors = {
        'fake_images' : fake_images,
        'fake_hidden' : z_dec,
        'like_loss' : like_loss,
        'eucd_loss' : eucd_loss,
        'norm_loss' : norm_loss,
    }

    return input_tensors, output_tensors
    

def inpainting_DLNet(options): 
    with tf.variable_scope('net_inpainting', reuse=options['reuse']):    

        batch_size = options['batch_size']
        image_shape = options['image_shape']        
        
        # Prepare inputs
        input_images = tf.placeholder('float32', [batch_size,] + image_shape, name='images')
        size_range = tf.placeholder('float32', [2,], name='inpaint_size_interval')
        shift_range = tf.placeholder('float32', [2,], name='inpaint_shift_interval')
        image_mask = generate_inpainting_mask(batch_size, image_shape, size_range, shift_range)
        Inputs = [input_images, size_range, shift_range, image_mask]

        # Prepare network weights
        feature_dims = [3, 64, 128, 256, 512]        
        Weights = generate_weights(feature_dims)
        
        # Prepare hyper-parameters for Adam
        VDlr_z, MDlr_z = 0, 0
        lr, beta1, beta2, gamma = 1e-3, 0.9, 0.999, 1e-16  

        # Recursively opt for a reasonable z_dec
        first_iter=True
        previous_z_dec=None 
        for i in xrange(options['k']):
            input_tensors, output_tensors = inpainting_net(Inputs, Weights,  
                first_iter, previous_z_dec, options['phase']=='Train')

            # Adam solver for maximizing the likelihood
            Dlr_z = tf.gradients(output_tensors['like_loss'], output_tensors['fake_hidden'])[0]
            MDlr_z = beta1*MDlr_z + (1-beta1)*Dlr_z    
            VDlr_z = beta2*VDlr_z + (1-beta2)*(Dlr_z*Dlr_z)    
            MDlr_z_cap = MDlr_z/(1-(beta1**(i+1)))        
            VDlr_z_cap = tf.maximum(VDlr_z/(1-(beta2**(i+1))), gamma)        
            previous_z_dec = output_tensors['fake_hidden'] - \
                (lr*MDlr_z_cap)/tf.sqrt(VDlr_z_cap)           
            first_iter=False

        return input_tensors, output_tensors


def interpolation_DLNet(options): 
    with tf.variable_scope('net_interpolation', reuse=options['reuse']):    

        batch_size = options['batch_size']
        image_shape = options['image_shape']        
        
        # Prepare inputs
        input_images = tf.placeholder('float32', [batch_size,] + image_shape, name='images')
        perc_range = tf.placeholder('float32', [2,], name='interpolation_perc')
        image_mask = generate_interpolation_mask(batch_size, image_shape, perc_range)
        Inputs = [input_images, perc_range, image_mask]

        # Prepare network weights
        feature_dims = [3, 64, 128, 256, 512]        
        Weights = generate_weights(feature_dims)
        
        # Prepare hyper-parameters for Adam
        VDlr_z, MDlr_z = 0, 0
        lr, beta1, beta2, gamma = 1e-2, 0.9, 0.999, 1e-16  

        # Recursively opt for a reasonable z_dec
        first_iter=True
        previous_z_dec=None 
        for i in xrange(options['k']):
            input_tensors, output_tensors = interpolation_net(Inputs, Weights,  
                first_iter, previous_z_dec, options['phase']=='Train')

            # Adam solver for maximizing the likelihood
            Dlr_z = tf.gradients(output_tensors['like_loss'], output_tensors['fake_hidden'])[0]
            MDlr_z = beta1*MDlr_z + (1-beta1)*Dlr_z    
            VDlr_z = beta2*VDlr_z + (1-beta2)*(Dlr_z*Dlr_z)    
            MDlr_z_cap = MDlr_z/(1-(beta1**(i+1)))        
            VDlr_z_cap = tf.maximum(VDlr_z/(1-(beta2**(i+1))), gamma)        
            previous_z_dec = output_tensors['fake_hidden'] - \
                (lr*MDlr_z_cap)/tf.sqrt(VDlr_z_cap)           
            first_iter=False

        return input_tensors, output_tensors