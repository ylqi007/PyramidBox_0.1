import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'


# =========================================================================== #
# General Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')
tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')
tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')
tf.app.flags.DEFINE_float(
    'gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')

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
    'batch_size', 4,
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

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    with tf.Graph().as_default():
        # Create global step.
        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.train.create_global_step()   # <tf.Variable 'global_step:0' shape=() dtype=int64_ref>

        # Get SSD network and its anchors.
        ssd_class = nets_factory.get_network(FLAGS.model_name)      # <class 'nets.ssd_vgg_300.SSDNet'>
        ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
        ssd_net = ssd_class(ssd_params)             # <nets.ssd_vgg_300.SSDNet object at 0x7fd5e709f198>
        ssd_shape = ssd_net.params.img_shape        # (300, 300) <class 'tuple'>
        ssd_anchors = ssd_net.anchors(ssd_shape)    # a list, each list contains 4 elements representing y, x, h, w

        # Select the preprocessing function.
        def _image_preprocessing_fn(_image_features):
            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name  # ssd_300_vgg <class 'str'>
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,
                                                                             is_training=True)
            image = _image_features['image']  # Tensor("IteratorGetNext:0", shape=(?, ?, ?), dtype=uint8, device=/device:CPU:0)
            shape = _image_features['shape']  # Tensor("IteratorGetNext:3", shape=(3,), dtype=int64, device=/device:CPU:0)
            glabels = _image_features['object/label']  # Tensor("IteratorGetNext:2", shape=(?,), dtype=int64, device=/device:CPU:0)
            gbboxes = _image_features['object/bbox']

            image, glabels, gbboxes = image_preprocessing_fn(image, glabels, gbboxes,
                                                             out_shape=(300, 300),
                                                             data_format='NCHW')
            return image, shape, glabels, gbboxes
            # _image_features['image'] = image
            # _image_features['shape'] = shape
            # _image_features['object/label'] = glabels
            # _image_features['object/bbox'] = gbboxes
            # return _image_features

        # Encode groundtruth labels and bboxes.
        def _bboxes_encode_fn(*image_features):
            _image, _shape, _glabels, _gbboxes = image_features
            _gclasses, _glocalisations, _gscores = ssd_net.bboxes_encode(_glabels, _gbboxes, ssd_anchors)
            return _image, _shape, _glabels, _gbboxes

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        # Select dataset.
        dataset = dataset_factory.get_dataset('pascalvoc_2007', 'train', '/tmp/pascalvoc_tfrecord/')
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_dataset'):
                dataset = dataset.map(_image_preprocessing_fn)
                dataset = dataset.map(_bboxes_encode_fn)
                dataset = dataset.batch(10)

            # Get for SSD network: image, labels, bboxes.
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            image_example = iterator.get_next()
            image, gclasses, glocalisations, gscores = image_example
            print('image: ', image)
            print('shape: ', gclasses)
            print('glabels: ', glocalisations)
            print('gbboxes: ', gscores)

            # # Encode groundtruth labels and bboxes.
            # print("##############################################################")
            # print("$$ Before encoding_fn glabels: ", glabels)   # shape=(?,), dtype=int64
            # print("$$ Before encoding_fn gbboxes: ", gbboxes)   # shape=(?, 4), dtype=float32
            # print("$$ Before encoding_fn ssd_anchors: ", type(ssd_anchors), len(ssd_anchors))
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # gclasses, glocalisations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
            # batch_shape = [1] + [len(ssd_anchors)] * 3
            # print("##############################################################")
            # print("$$ After Encode groundtruth labels and bboxes -- gclasses: ", gclasses)  # list with len=6
            # print("$$ After Encode groundtruth labels and bboxes -- glocalisations: ", glocalisations)  # list with len=6
            # print("$$ After Encode groundtruth labels and bboxes -- gscores: ", gscores)    # list with len=6
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            #
            # # Training batches and queue.
            # r1 = tf_utils.reshape_list([image, gclasses, glocalisations, gscores])
            # print('# r1: ', r1)
            # r = tf.train.batch(tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),   # Flatten everything
            #                    batch_size=FLAGS.batch_size,
            #                    num_threads=FLAGS.num_preprocessing_threads,
            #                    capacity=5 * FLAGS.batch_size)
            # print("##############################################################")
            # print("r: ", r)
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            # b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(r, batch_shape)
            # print('b_image', b_image)
            # print('b_gclasses', b_gclasses)
            # print('b_glocalisations', b_glocalisations)
            # print('b_gscores', b_gscores)

            # Intermediate queueing: unique batch computation pipeline for all
            # GPUs running the training.
            # batch_queue = slim.prefetch_queue.prefetch_queue(
            #     tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]),
            #     capacity=2 * deploy_config.num_clones)

        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #

        i = 0
        with tf.compat.v1.Session() as sess:
            try:
                # while True:
                for _ in range(1):
                    # test = sess.run(b_gclasses)  # after sess, test is <class 'numpy.ndarray'>
                    test = gclasses
                    print(type(test), test.shape)
                    print("#!#", i, test)
                    i = i + 1
            except tf.errors.OutOfRangeError:
                print("End! totally: ", i)


if __name__ == '__main__':
    tf.compat.v1.app.run()
