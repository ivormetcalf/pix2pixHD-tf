import tensorflow as tf
import numpy as np
import os
import datetime
import models
from process_input import create_traineval_dataset, create_inference_dataset
from PIL import Image
from options import Options

opt = Options().opt
timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d--%Hh%Mm%Ss")

def get_model_paths(d):
    ''' Given a directory containing saved models, returns a dict of paths to each model. '''
    path_dict = {}
    path_dict["coarse_generator"] = os.path.join(d, "coarse_generator.h5")
    path_dict["fine_generator"] = os.path.join(d, "fine_generator.h5")
    for i in range(3):
        disc_name = "discriminator%d" % i
        path_dict[disc_name] = os.path.join(d, disc_name + ".h5")
    return path_dict

def get_load_dir(model_dir, subpath):
    def get_latest(directory, numerical=False):
        # Returns subfolder with 'latest' name in directory.
        if not tf.gfile.IsDirectory(directory):
            return None
        dirs = tf.gfile.ListDirectory(directory)
        dirs = [x for x in dirs if tf.gfile.IsDirectory(os.path.join(directory, x))]
        if len(dirs) == 0:
            return None

        if numerical:   # Sort numerically instead of lexicographically
            dir_idx = [int(x) for x in dirs]
            dirs = sorted(zip(dir_idx, dirs))
            dirs = [x[1] for x in dirs]
        else:
            dirs = sorted(dirs)
        return os.path.join(directory, dirs[len(dirs)-1])

    if subpath == 'False':  # Do not load a checkpoint
        return None
    elif subpath == None:   # Load the latest checkpoint of the latest run
        latest_model = get_latest(model_dir)
        if latest_model == None:
            return None
        else:
            return get_latest(latest_model, numerical=True)
    else:
        # Conveniently strip the model_dir prefix from subpath when present
        if subpath.startswith('%s/' % model_dir):
            subpath = subpath[len(model_dir) + 1:]
            
        if os.path.dirname(subpath) == '':  # Load the latest checkpoint of the specified run
            return get_latest(os.path.join(model_dir, subpath), numerical=True)
        else:
            return os.path.join(model_dir, subpath)     # Load the specified run and checkpoint

def load_all_models(model_dict, force_load=False):
    def load_model(model, path):    # Returns true if weights were successfully loaded.
        try:
            model.load_weights(path)
            return True
        except:
            return False

    # Load all models
    for name, model in model_dict.items():
        if opt.load_model_paths != None and load_model(model, opt.load_model_paths[name]):
            print("Successfully reloaded model: %s, from %s" % (name, opt.load_model_paths[name]))
        else:
            print("Using freshly initialized model: %s" % name)
            assert(force_load == False)     # Inference mode needs trained models.

def train(training_dataset, eval_dataset):
    tf.reset_default_graph()

    inputs, outputs, model_dict = models.define_model(opt)

    iterator = training_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    eval_iterator = eval_dataset.make_one_shot_iterator()
    next_eval_element = eval_iterator.get_next()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    load_all_models(model_dict)

    writer = tf.summary.FileWriter(opt.log_dir, sess.graph)

    if opt.trace:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = run_metadata = None

    for training_steps in range(opt.steps):
        try:
            label_imgs, target_imgs = sess.run(next_element)
        except tf.errors.OutOfRangeError: 
            break

        if opt.mode == 'train':

            ops = ['disc_optimizer']
            if 'coarse' in opt.phase:
                ops.append('coarse_optimizer')
            if 'fine' in opt.phase:
                ops.append('fine_optimizer') 
            
            if training_steps % opt.summary_freq == 0:
                ops.append('summary')

            op_dict = {o: outputs[o] for o in ops}

            feed_dict = {inputs['labels']: label_imgs, inputs['images']: target_imgs, inputs['lr']: opt.lr}
            fetches = sess.run(op_dict, feed_dict, options=run_options, run_metadata=run_metadata)

            if opt.trace:
                writer.add_run_metadata(run_metadata, 'step%d' % training_steps)

            if training_steps % opt.summary_freq == 0:                
                writer.add_summary(fetches['summary'], training_steps)
            
            if opt.save_freq != 0 and training_steps != 0 and training_steps % opt.save_freq == 0:
                save_dir = os.path.join(opt.model_dir, timestamp_str, str(training_steps))
                save_paths = get_model_paths(save_dir)
                tf.gfile.MakeDirs(save_dir)

                # Save all models
                for name, model in model_dict.items():
                    model.save_weights(save_paths[name])

        if training_steps % opt.eval_freq == 0 or opt.mode == 'eval':
            eval_labels, eval_images = sess.run(next_eval_element)
            image_summary = sess.run(outputs['image_summary'], feed_dict={inputs['labels']: eval_labels, inputs['images']: eval_images})
            writer.add_summary(image_summary, training_steps)

            print("Training Step %d, %.1f%%" % (training_steps, training_steps*100.0/opt.steps))

def inference(inference_dataset):
    tf.reset_default_graph()

    inputs, outputs, model_dict = models.define_model(opt)

    iterator = inference_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    load_all_models(model_dict, force_load=True)

    while True:
        try:
            label_imgs, label_filenames = sess.run(next_element)
        except tf.errors.OutOfRangeError: 
            break

    scale = 0 if 'fine' in opt.phase else 1
    out_imgs = sess.run(outputs['output_scales'][scale], feed_dict={inputs['labels']: label_imgs})
    
    out_imgs = (out_imgs + 1) / 2 * 255
    out_imgs = np.array(out_imgs, dtype=np.uint8)

    for i in range(opt.batch_size):
        img_name = str(label_filenames[i])
        img_name = os.path.splitext(img_name)[0] + '.' + opt.output_ext
        
        img = Image.fromarray(out_imgs[i])
        img = img.save(os.path.join(opt.output_dir, img_name))
    

if __name__ == '__main__':

    if opt.seed is None:
        opt.seed = np.random.randint(0, 2**31 - 1)

    tf.set_random_seed(opt.seed)
    np.random.seed(opt.seed)

    opt.log_dir = os.path.join("tensorboard", timestamp_str)
    opt.load_dir = get_load_dir(opt.model_dir, opt.load)
    opt.load_model_paths = None if opt.load_dir is None else get_model_paths(opt.load_dir)

    if opt.output_dir == None:
        opt.output_dir = "output"

    image_size = opt.image_dim
    if opt.phase == 'coarse':
        image_size = [opt.image_dim[0] // 2, opt.image_dim[1] // 2]

    if opt.mode == "train":
        tf.gfile.MakeDirs(opt.log_dir)
        train_dataset, eval_dataset = create_traineval_dataset(opt.data_dir, image_size, opt.batch_size, opt.use_instmaps)
        train(train_dataset, eval_dataset)
    else:
        inference_dataset = create_inference_dataset(opt.data_dir, image_size, opt.batch_size, opt.use_instmaps)
        tf.gfile.MakeDirs(opt.output_dir)
        inference(inference_dataset)
