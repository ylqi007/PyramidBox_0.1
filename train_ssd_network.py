import tensorflow as tf
from datasets import dataset_factory

slim = tf.contrib.slim


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

    with tf.Graph().as_default():
        # Create global step.
        with tf.device('/cpu:0'):
            global_step = tf.compat.v1.train.create_global_step()   # <tf.Variable 'global_step:0' shape=() dtype=int64_ref>

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
            gbboxes = image_example['object/bbox']

        i = 0
        with tf.compat.v1.Session() as sess:
            try:
                # while True:
                for _ in range(1):
                    test = sess.run(shape)  # after sess, test is <class 'numpy.ndarray'>

                    print(type(test))
                    print("##", i, test)
                    i = i + 1
                # with tf.device('/cpu:0'):
                #     # for _ in range(10):
                #     while True:
                #         image = sess.run(image_example)
                #         print("##", i, image)
                #         i = i + 1
            except tf.errors.OutOfRangeError:
                print("End! totally: ", i)


if __name__ == '__main__':
    tf.compat.v1.app.run()
