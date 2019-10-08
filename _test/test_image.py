

import tensorflow as tf
import matplotlib.pyplot as plt

raw_image_data = tf.gfile.GFile('000001.jpg', mode='rb').read()     # <class 'bytes'>
print(type(raw_image_data))
print(raw_image_data)

with tf.Session() as sess:
    image = tf.image.decode_jpeg(raw_image_data)
    # print('image', type(image), image)
    #
    boxes = tf.constant([[[240/500, 48/353, 371/500, 195/353], [12/500, 8/353, 498/500, 352/353]]])
    # print('boxes: ', type(boxes), boxes)
    batched_image = tf.expand_dims(tf.image.convert_image_dtype(image, tf.float32), 0)
    # print('batched_image: ', type(batched_image), batched_image)
    #
    # image_with_box = tf.image.draw_bounding_boxes(batched_image, boxes)
    # plt.imshow(image_with_box[0].eval())
    # plt.show()

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(tf.shape(image),
                                                                        bounding_boxes=boxes,
                                                                        min_object_covered=0.5)
    distored_image = tf.slice(image, begin, size)
    image_with_box = tf.image.draw_bounding_boxes(batched_image, bbox_for_draw)
    print(type(distored_image), distored_image)
    print(type(image_with_box[0]), image_with_box[0])
    print(bbox_for_draw.eval())
    res1, res2 = sess.run([image_with_box[0], distored_image])
    plt.subplot(1, 2, 1)
    plt.title("image with a random box")
    plt.imshow(res1)
    plt.subplot(1, 2, 2)
    plt.title("destorted image")
    plt.imshow(res2)
    plt.show()

