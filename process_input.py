import tensorflow as tf
import os

def get_all_files(files_dir):
    return tf.gfile.Glob(os.path.join(files_dir, "*"))

def get_training_data_folders(use_instmaps):
    data_folders = ["labels", "targets"]
    if use_instmaps:
        data_folders.append("instmaps")
    return [[x+y for y in data_folders] for x in ['train_', 'eval_']]

def decode_image(image_encoded, channels=3):
    # decode_jpeg works for both png and jpeg. decode_image does not work with resize_images.
    image_decoded = tf.image.decode_jpeg(image_encoded, channels=channels)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    return image_decoded

def resize_and_rescale(image, image_size):
    scaled_shape = tf.convert_to_tensor([image_size[1], image_size[0]])
    image_scaled = tf.image.resize_images(image, scaled_shape)
    image_scaled = image_scaled * 2 - 1
    return image_scaled

def get_instmap_edges(t):
    p = tf.pad(t, [[1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    pads = [p[:-2,1:-1,:], 
            p[2:,1:-1,:], 
            p[1:-1,:-2,:], 
            p[1:-1,2:,:]]
    e = tf.ones_like(t, dtype=tf.bool)
    for x in pads:
        e = tf.logical_and(e, tf.equal(t, x))
    return tf.cast(e, tf.float32)

def create_dataset(files, image_folder, image_dim, batch_size, use_instmaps, shuffle_size):
    if image_folder:
        if use_instmaps:
            files = [[x, y, z] for x, y, z in zip(files[0], files[1], files[2])]
        else:
            files = [[x, y] for x, y in zip(files[0], files[1])]

    assert(len(files) > 0)
    print(files)

    dataset = tf.data.Dataset.from_tensor_slices(files)

    if not image_folder:
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))

    dataset = dataset.shuffle(buffer_size=shuffle_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(4)

    def tfrecord_parser(record):
        keys_to_features = {
            "real/filename": tf.FixedLenFeature((), tf.string, default_value=""),
            "real/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "pose/filename": tf.FixedLenFeature((), tf.string, default_value=""),
            "pose/encoded": tf.FixedLenFeature((), tf.string, default_value="")
        }
        if use_instmaps:
            keys_to_features["instmap/filename"] = tf.FixedLenFeature((), tf.string, default_value="")
            keys_to_features["instmap/encoded_img"] = tf.FixedLenFeature((), tf.string, default_value="")
        
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        label_image = resize_and_rescale(decode_image(parsed["pose/encoded"]), image_dim)
        target_image = resize_and_rescale(decode_image(parsed["real/encoded"]), image_dim)
        if use_instmaps:
            instmap_processed = decode_image(parsed["instmap/encoded_img"], 1)
            instmap_processed = get_instmap_edges(instmap_processed)
            instmap_processed = resize_and_rescale(instmap_processed, image_dim)
            label_image = tf.concat([label_image, instmap_processed], axis=3)

        return label_image, target_image

    def image_folder_parser(filenames):
        label_image = resize_and_rescale(decode_image(tf.read_file(filenames[0])), image_dim)
        target_image = resize_and_rescale(decode_image(tf.read_file(filenames[1])), image_dim)

        if use_instmaps:
            instmap_filename = filenames[2]
            instmap_processed = decode_image(tf.read_file(instmap_filename), 1)
            instmap_processed = get_instmap_edges(instmap_processed)
            instmap_processed = resize_and_rescale(instmap_processed, image_dim)
            label_image = tf.concat([label_image, instmap_processed], axis=2)

        return label_image, target_image

    if image_folder:
        dataset = dataset.map(image_folder_parser, num_parallel_calls=4)
    else:
        dataset = dataset.map(tfrecord_parser, num_parallel_calls=4)

    dataset = dataset.batch(batch_size)
    return dataset


def create_traineval_dataset(dataset_dir, image_dim, batch_size, use_instmaps, default_shuffle=1000000):
    # Detect if data is in image folder or tfrecord format
    # TF Record - 2 folders called 'training_data' and 'eval_data'
    # Image Folder - train_labels, train_targets, eval_labels, eval_targets, [train_instmaps, eval_instmaps]
    dir_elems = [x for x in tf.gfile.ListDirectory(dataset_dir) if tf.gfile.IsDirectory(os.path.join(dataset_dir, x))]
    dir_elems = [x.rstrip('/') for x in dir_elems]  # On GCS directories have '/' at the end.
    if 'training_data' in dir_elems and 'eval_data' in dir_elems:
        image_folder = False
    else:
        image_folder = True

    if image_folder:
        print("Loading images from: '%s'" % dataset_dir)
        data_folders = get_training_data_folders(use_instmaps)
        training_files, eval_files = [[get_all_files(os.path.join(dataset_dir, y)) for y in x] for x in data_folders]
        
        training_dataset = create_dataset(training_files, True, image_dim, batch_size, use_instmaps, len(training_files[0]))
        eval_dataset = create_dataset(eval_files, True, image_dim, batch_size, use_instmaps, len(eval_files[0]))
    else:
        print("Loading tfrecords from: '%s'" % dataset_dir)
        training_files = get_all_files(os.path.join(dataset_dir, 'training_data'))
        eval_files = get_all_files(os.path.join(dataset_dir, 'eval_data'))

        training_dataset = create_dataset(training_files, False, image_dim, batch_size, use_instmaps, default_shuffle)
        eval_dataset = create_dataset(eval_files, False, image_dim, batch_size, use_instmaps, default_shuffle)

    return training_dataset, eval_dataset

def create_inference_dataset(dataset_dir, image_dim, batch_size, use_instmaps):
    # Detect if data is in image folder or tfrecord format
    image_folder = False
    filenames = tf.gfile.Glob(os.path.join(dataset_dir, '*.tfrecord'))
    if len(filenames) == 0:
        image_folder = True
        if use_instmaps:        
            label_files = get_all_files(os.path.join(dataset_dir, 'instmaps'))
            instmap_files = get_all_files(os.path.join(dataset_dir, 'instmaps'))
            filenames = zip(label_files, instmap_files)
        else:
            filenames = get_all_files(dataset_dir)

    print(filenames)
    assert(len(filenames) > 0)

    def tfrecord_parser(record):
        keys_to_features = {
            "real/filename": tf.FixedLenFeature((), tf.string, default_value=""),
            "real/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
            "pose/filename": tf.FixedLenFeature((), tf.string, default_value=""),
            "pose/encoded": tf.FixedLenFeature((), tf.string, default_value="")
        }
        if use_instmaps:
            keys_to_features["instmap/filename"] = tf.FixedLenFeature((), tf.string, default_value="")
            keys_to_features["instmap/encoded_img"] = tf.FixedLenFeature((), tf.string, default_value="")
        
        parsed = tf.parse_single_example(record, keys_to_features)

        # Perform additional preprocessing on the parsed data.
        label_image = resize_and_rescale(decode_image(parsed["pose/encoded"]), image_dim)
        if use_instmaps:
            instmap_processed = decode_image(parsed["instmap/encoded_img"], 1)
            instmap_processed = get_instmap_edges(instmap_processed)
            instmap_processed = resize_and_rescale(instmap_processed, image_dim)
            label_image = tf.concat([label_image, instmap_processed], axis=3)

        return label_image, parsed["pose/filename"]

    def image_folder_parser(filenames):
        label_filename = filenames[0] if use_instmaps else filenames
        label_image = resize_and_rescale(decode_image(tf.read_file(label_filename)), image_dim)
        if use_instmaps:
            label_filename = filenames[0]
            instmap_filename = filenames[1]
            instmap_processed = decode_image(tf.read_file(instmap_filename), 1)
            instmap_processed = get_instmap_edges(instmap_processed)
            instmap_processed = resize_and_rescale(instmap_processed, image_dim)
            label_image = tf.concat([label_image, instmap_processed], axis=2)
        return label_image, label_filename

    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if image_folder:
        dataset = dataset.map(image_folder_parser)
    else:
        dataset = dataset.apply(tf.data.experimental.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))
        dataset = dataset.map(tfrecord_parser)

    dataset = dataset.batch(batch_size)

    return dataset