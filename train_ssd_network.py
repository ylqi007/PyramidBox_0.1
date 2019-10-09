import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet',
    'The name of the dataset to load.')
tf.app.flags.DEFINE_integer(
    'num_classes', 21,
    'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train',
    'The name of the train/test split.')
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')
tf.app.flags.DEFINE_string(
    'model_name', 'ssd_300_vgg',
    'The name of the architecture to train.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None,
    'The name of the preprocessing to use.'
    ' If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32,
    'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'train_image_size', None,
    'Train image size')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None,
    'The maximum number of training steps.')


FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    with tf.Graph().as_default():
        # Create global step.
        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.train.create_global_step()   # <tf.Variable 'global_step:0' shape=() dtype=int64_ref>

        # Get SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)
        ssd_params = ssd_class.default_params._replace(num_classes=10)
        ssd_net = ssd_class(ssd_params)
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)    # a list, each list contains 4 elements representing y, x, h, w

        # Select the preprocessing function.
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,
                                                                         is_training=True)

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        # Select dataset.
        dataset = dataset_factory.get_dataset('pascalvoc_2007', 'train', '/tmp/pascalvoc_tfrecord/')
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_dataset'):
                dataset = dataset.shuffle(10)

            # Get for SSD network: image, labels, bboxes.
            print("dataset:\n", dataset)
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            image_example = iterator.get_next()
            image = image_example['image']
            shape = image_example['shape']
            glabels = image_example['object/label']
            gbboxes = tf.transpose(image_example['object/bbox'])
            print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
            print('## image: ', image)
            print('## shape: ', shape)
            print('## glabels: ', glabels)
            print('## gbboxes: ', gbboxes)

            # Pre-processing image, labels and bboxes.
            image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,
                                                             out_shape=ssd_shape,
                                                             data_format=DATA_FORMAT)
            print("##############################################################")
            print("$$ After preprocess_fn image: ", image)
            print("$$ After preprocess_fn glabels: ", glabels)
            print("$$ After preprocess_fn gbboxes: ", gbboxes)

        i = 0
        with tf.compat.v1.Session() as sess:
            try:
                # while True:
                for _ in range(1):
                    test = sess.run(gbboxes)  # after sess, test is <class 'numpy.ndarray'>

                    print(type(test), test.shape)
                    print("##", i, test)
                    i = i + 1
            except tf.errors.OutOfRangeError:
                print("End! totally: ", i)


if __name__ == '__main__':
    tf.compat.v1.app.run()
