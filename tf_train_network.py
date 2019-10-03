import tensorflow as tf
from datasets import dataset_factory


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main(_):

    with tf.Graph().as_default():
        dataset = dataset_factory.get_dataset('pascalvoc_2007', 'train', '/tmp/pascalvoc_tfrecord/')
        dataset = dataset.batch(1)
        print("dataset:\n", dataset)

        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
        image_example = iterator.get_next()

        i = 0
        with tf.compat.v1.Session() as sess:
            try:
                while True:
                    image = sess.run(image_example)
                    print("##", i, image)
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
