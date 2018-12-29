import tensorflow as tf
import numpy as np

# Weight Initializers
conv_init = tf.random_normal_initializer(0, 0.02)
batchnorm_init = tf.random_normal_initializer(1.0, 0.02)

# Applies reflection padding to an image_batch
def reflection_pad(image_batch, pad):
    paddings = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
    return tf.keras.layers.Lambda(lambda x: tf.pad(x, paddings, mode='REFLECT'))(image_batch)

# A thin wrapper around conv2D that applies reflection_padding when required
def conv2D(img, k, f, s, reflect_pad):
    pad_mode = 'valid' if reflect_pad else 'same'
    if reflect_pad: 
        img = reflection_pad(img, f//2)
    return tf.keras.layers.Conv2D(k, (f, f), strides=(s, s), padding=pad_mode, kernel_initializer=conv_init)(img)

# Downsamples the image i times.
def downsample(img, i):
    return tf.layers.average_pooling2d(img, 2 ** i, 2 ** i)

def c7s1(x, k, activation, reflect_pad=True):    # 7×7 Convolution-InstanceNorm-Activation with k filters
    x = conv2D(x, k, 7, 1, reflect_pad=reflect_pad)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=batchnorm_init)(x, training=True)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def d(x, k, reflect_pad=True):       # 3×3 Convolution-InstanceNorm-ReLU with k filters
    x = conv2D(x, k, 3, 2, reflect_pad=reflect_pad)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=batchnorm_init)(x, training=True)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x

def R(x, k, reflect_pad=True):       # Residual block with 2 3×3 Convolution layers with k filters.
    y = x
    y = conv2D(y, k, 3, 1, reflect_pad=reflect_pad)
    y = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=batchnorm_init)(y, training=True)
    y = tf.keras.layers.LeakyReLU(alpha=0.2)(y)
    y = conv2D(y, k, 3, 1, reflect_pad=reflect_pad)
    y = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=batchnorm_init)(y, training=True)
    y = tf.keras.layers.Add()([x, y])
    return y

def u(x, k):       # 3×3 Transposed Convolution-InstanceNorm-ReLU layer with k filters.
    x = tf.keras.layers.Conv2DTranspose(k, (3, 3), strides=(2, 2), padding='same', kernel_initializer=conv_init)(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=batchnorm_init)(x, training=True)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    return x


# Loss Components

def discriminator_loss(real_output, generated_output, lsgan):
    if lsgan:
        real_loss = tf.losses.mean_squared_error(tf.ones_like(real_output), real_output)
        generated_loss = tf.losses.mean_squared_error(tf.zeros_like(generated_output), generated_output)
    else:
        real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
        generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss
    return total_loss

def generator_gan_loss(generated_output, lsgan):
    if lsgan:
        return tf.losses.mean_squared_error(tf.ones_like(generated_output), generated_output)
    else:
        return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

def generator_feature_matching_loss(gen_activations, real_activations):
    total_loss = 0
    for g, r in zip(gen_activations, real_activations):
        num_elements = np.prod(g.shape.as_list()[1:])         # Number of elements excluding batch dimension
        total_loss += tf.losses.absolute_difference(g, r) / num_elements    # Paper suggests L1 loss
    return total_loss


# Generators

def define_global_generator(input_label_shape, output_channels, reflection_padding=True):
    ''' Define the coarse 'global' generator. '''

    down_layers = [64, 128, 256, 256]
    residual_layers = [256] * 9
    up_layers = [128, 128, 64, 32]

    input_label = tf.keras.Input(shape=input_label_shape)

    result = c7s1(input_label, 16, 'relu', reflect_pad=reflection_padding)
    
    for k in down_layers:
        result = d(result, k, reflect_pad=reflection_padding)
    for k in residual_layers:
        result = R(result, k, reflect_pad=reflection_padding)
    for k in up_layers:
        result = u(result, k)

    last_feature_map = result
    result = c7s1(result, output_channels, 'tanh', reflect_pad=reflection_padding)
    
    return tf.keras.Model(inputs=input_label, outputs=[result, last_feature_map])

def define_enhancer_generator(input_label_shape, coarse_input_shape, output_channels, reflection_padding=True):
    ''' Define the fine 'enhancer' generator. '''

    input_label = tf.keras.Input(shape=input_label_shape)    
    coarse_feature_map = tf.keras.Input(shape=coarse_input_shape)

    residual_layers = [32] * 3

    result = c7s1(input_label, 16, 'relu', reflect_pad=reflection_padding)
    result = d(result, 32, reflect_pad=reflection_padding)
    result = tf.keras.layers.Add()([result, coarse_feature_map])
    for k in residual_layers:
        result = R(result, k, reflect_pad=reflection_padding)
    result = u(result, 16)
    result = c7s1(result, output_channels, 'tanh', reflect_pad=reflection_padding)

    return tf.keras.Model(inputs=[input_label, coarse_feature_map], outputs=result)


# Discriminators (must return the list of outputs of each layer for fm loss)

def define_patch_discriminator(label_shape, target_shape):
    
    def conv(units):
        initializer = tf.random_normal_initializer(0, 0.02)
        return tf.keras.layers.Conv2D(units, (4, 4), strides=(2, 2), padding='valid', kernel_initializer=initializer)

    def batchnorm():
        initializer = tf.random_normal_initializer(1.0, 0.02)
        return tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1, gamma_initializer=initializer)

    def leaky_relu():
        return tf.keras.layers.LeakyReLU(alpha=0.2)

    conv_units = [128, 256, 512]
    label_img = tf.keras.Input(shape=label_shape)
    target_img = tf.keras.Input(shape=target_shape)

    result = tf.keras.layers.Concatenate()([label_img, target_img])

    layers = []

    result = conv(64)(result)
    result = leaky_relu()(result)
    layers.append(result)
    
    for k in conv_units:
        result = conv(k)(result)
        result = batchnorm()(result)
        result = leaky_relu()(result)
        layers.append(result)
    
    result = conv(1)(result)
    result = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.reduce_mean(tf.layers.flatten(x), 1), (-1, 1)))(result)
    layers.append(result)

    return tf.keras.Model(inputs=[label_img, target_img], outputs=layers)


def define_model(opt):
    ''' Defines the pix2pixHD model. Returns inputs, outputs and a dict of all the keras models used. 
        The dict is used for loading/saving models and hence the keras models must contain all trainable parameters. '''

    # Scale 0 - Max Resolution, Scale 1 - Half Resolution, Scale 2 - Quarter Resolution
    start_scale = 1 if opt.phase == 'coarse' else 0

    label_channels = 4 if opt.use_instmaps else 3
    target_channels = 3

    label_shapes = []
    target_shapes = []
    for i in range(3):
        # Image format: (Height, Width, Channels)
        s = (opt.image_dim[1] // (2 ** i), opt.image_dim[0] // (2 ** i))
        
        label_shapes.append(s + (label_channels,))
        target_shapes.append(s + (target_channels,))
    
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    real_scales = [None] * 3
    label_scales = [None] * 3
    gen_scales = [None] * 3
    discriminators = [None] * 3
    coarse_image = coarse_feature_map = fine_image = None

    for i in range(start_scale, 3):
        with tf.name_scope("scale_%d" % (i)):
            if i == start_scale:    # Define placeholders
                input_real = tf.placeholder(tf.float32, (None,) + target_shapes[start_scale], name="real")
                input_label = tf.placeholder(tf.float32, (None,) + label_shapes[start_scale], name="label")
                real_scales[i] = input_real
                label_scales[i] = input_label
            else:                   # Create downsampled scales
                with tf.name_scope("real"):
                    real_scales[i] = downsample(real_scales[start_scale], i-start_scale)
                with tf.name_scope("label"):
                    label_scales[i] = downsample(label_scales[start_scale], i-start_scale)

    # Create coarse generator
    with tf.name_scope("coarse_generator"):
        coarse_generator = define_global_generator(label_shapes[1], target_channels, reflection_padding=opt.reflect_padding)

    # Create coarse image
    coarse_image, coarse_feature_map = coarse_generator(label_scales[1], training=True)
    gen_scales[1] = coarse_image

    if 'fine' in opt.phase:
        # Create fine generator
        with tf.name_scope("fine_generator"):
            shape2 = coarse_generator.output_shape[1][1:]
            fine_generator = define_enhancer_generator(label_shapes[0], shape2, target_channels, reflection_padding=opt.reflect_padding)

        # Create fine image
        fine_image = fine_generator([label_scales[0], coarse_feature_map], training=True)
        gen_scales[0] = fine_image
    else:
        fine_generator = None

    for i in range(start_scale+1, 3):
        gen_scales[i] = downsample(gen_scales[start_scale], i-start_scale)

    generator_total_loss = 0
    discriminator_total_loss = 0

    # Discriminate each scale
    for i in range(start_scale, 3):
        with tf.name_scope("scale_%d" % i):
            # Define discriminators
            with tf.name_scope("discriminator"):
                disc = define_patch_discriminator(label_shapes[i], target_shapes[i])
                discriminators[i] = disc

            def get_losses(generated_img):
                with tf.name_scope("real_activations"):
                    real_activations = disc([real_scales[i], label_scales[i]], training=True)
                with tf.name_scope("gen_activations"):
                    gen_activations = disc([generated_img, label_scales[i]], training=True)
                
                with tf.name_scope("generator_gan_loss"):
                    gen_gan_loss = generator_gan_loss(gen_activations[-1], lsgan=opt.lsgan)
                with tf.name_scope("generator_fm_loss"):
                    gen_fm_loss = generator_feature_matching_loss(gen_activations, real_activations)

                with tf.name_scope("discriminator_loss"):
                    disc_loss = discriminator_loss(real_activations[-1], gen_activations[-1], lsgan=opt.lsgan)

                return gen_gan_loss, gen_fm_loss, disc_loss

            losses = get_losses(gen_scales[i])
            generator_total_loss += losses[0] * opt.gan_weight
            generator_total_loss += losses[1] * opt.fm_weight
            discriminator_total_loss += losses[2]

    optim = tf.train.AdamOptimizer(learning_rate, 0.5)

    # Apparently this is needed or batch_norm parameters will not update when optimizing
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        coarse_gen_optimizer = fine_gen_optimizer = None

        if 'coarse' in opt.phase:
            coarse_gen_optimizer = optim.minimize(generator_total_loss, var_list=coarse_generator.variables)
        if 'fine' in opt.phase:
            fine_gen_optimizer = optim.minimize(generator_total_loss, var_list=fine_generator.variables)

        disc_variables = [v for d in discriminators if d is not None for v in d.variables]
        disc_optimizer = optim.minimize(discriminator_total_loss, var_list=disc_variables)
    
    # Summaries
    scalar_summaries = [tf.summary.scalar('generator_total_loss', generator_total_loss),
                        tf.summary.scalar('discriminator_total_loss', discriminator_total_loss)]

    image_summaries = [tf.summary.image('real', input_real), tf.summary.image('coarse', coarse_image)]
    if opt.use_instmaps:
        image_summaries += [tf.summary.image('label', input_label[:, :, :, :-1]), 
                            tf.summary.image('edge_map', input_label[:, :, :, -1:])]
    else:
        image_summaries.append(tf.summary.image('label', input_label))

    if 'fine' in opt.phase:
        image_summaries.append(tf.summary.image('fine', fine_image))

    inputs = {  'images': input_real, 
                'labels': input_label, 
                'lr': learning_rate}

    outputs = { 'output_scales': gen_scales,
                'coarse_optimizer': coarse_gen_optimizer,
                'fine_optimizer': fine_gen_optimizer,
                'disc_optimizer': disc_optimizer,
                'summary': tf.summary.merge(scalar_summaries),
                'image_summary': tf.summary.merge(image_summaries)}

    
    # Create a dict containing all keras models (for saving/loading).

    model_dict = {'coarse_generator': coarse_generator}
    if 'fine' in opt.phase:
        model_dict['fine_generator'] = fine_generator

    for i, d in enumerate(discriminators):
        if d == None:
            continue
        disc_name = 'discriminator%d' % i
        model_dict[disc_name] = d

    return inputs, outputs, model_dict
