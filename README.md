# PyramidBox_0.1
* Python==3.6.8
* TensorFlow==1.14.0

## Loading data from TFRecord files
### Reading from TFRecord files
```python
raw_image_dataset = tf.data.TFRecordDataset(filenames)

def _parse_example_function(example_proto):
    image_features = tf.io.parse_single_example(example_proto, items_to_descriptions)
    image_decode = tf.io.decode_jpeg(image_features['image/raw_data'])
    image_features['image/raw_data'] = image_decode

parse_image_dataset = raw_image_dataset.map(_parse_example_function)
```
The elements in `raw_dataset` are serialized string;
then parse each example proto to get parsed example.

20191004: add handle_features part

20191005: todo
1. preprocessing images
2. how to batch images

20191010:
- [x] `image_preprocessing_fn(image, glables, gbboxes, output_shape, data_format)`
- [x] `ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)`
- [ ] `tf.train.batch()`
- [ ] `display b_image and feature maps after conv layers`
