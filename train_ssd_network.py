import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import tf_utils

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'

# =========================================================================== #
# SSD Network flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float(
    'negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float(
    'match_threshold', 0.5, 'Matching threshold in the loss function.')

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
# Optimization Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# =========================================================================== #
# Learning Rate Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

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

# =========================================================================== #
# Fine-Tuning Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_model_scope', None,
    'Model scope in the checkpoint. None if the same as the trained model.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model_deploy. Keep TF Slim Models structure.
        # Useful if want to need multiple GPUs and/or servers in the future.
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=0,
            num_replicas=1,
            num_ps_tasks=0)

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
                                                             out_shape=ssd_shape,
                                                             data_format=DATA_FORMAT)
            image_features = {'image': image,
                              'shape': shape,
                              'object/label': glabels,
                              'object/gbboxes': gbboxes}
            return image_features       # A dictionary

        # Encode groundtruth labels and bboxes.
        def _bboxes_encode_fn(image_features, anchors):
            _image = image_features['image']
            _shape = image_features['shape']
            _glabels = image_features['object/label']
            _gbboxes = image_features['object/gbboxes']

            _gclasses, _glocalisations, _gscores = ssd_net.bboxes_encode(_glabels, _gbboxes, anchors)

            _flatten_image_features = tf_utils.reshape_list([_image, _gclasses, _glocalisations, _gscores])
            # print('\n##############################################################')
            # print('In encode function, after tf_utils.reshape_list')
            # print(_flatten_image_features)
            return _flatten_image_features

        # =================================================================== #
        # Create a dataset provider and batches.
        # =================================================================== #
        # Select dataset.
        dataset = dataset_factory.get_dataset('pascalvoc_2007', 'train', '/tmp/pascalvoc_tfrecord/')
        with tf.device('/cpu:0'):
            with tf.name_scope(FLAGS.dataset_name + '_dataset'):
                dataset = dataset.map(_image_preprocessing_fn)
                dataset = dataset.map(lambda x: _bboxes_encode_fn(x, ssd_anchors))
                dataset = dataset.batch(32)

            # Get for SSD network: image, labels, bboxes.
            iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
            batch_example = iterator.get_next()
            batch_shape = [1] + [len(ssd_anchors)] * 3
            # b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(batch_example, batch_shape)

        # =================================================================== #
        # Define the model running on every GPU.
        # =================================================================== #
        def clone_fn(batch_example):
            b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(batch_example, batch_shape)

            # Construct SSD network
            arg_scope = ssd_net.arg_scope(weight_decay=FLAGS.weight_decay,
                                          data_format=DATA_FORMAT)

            with slim.arg_scope(arg_scope):
                predictions, localisations, logits, end_points = ssd_net.net(b_image,
                                                                             is_training=True)

            ssd_net.losses(logits, localisations,
                           b_gclasses, b_glocalisations, b_gscores,
                           match_threshold=FLAGS.match_threshold,
                           negative_ratio=FLAGS.negative_ratio,
                           alpha=FLAGS.loss_alpha,
                           label_smoothing=FLAGS.label_smoothing)
            return end_points


if __name__ == '__main__':
    tf.compat.v1.app.run()
