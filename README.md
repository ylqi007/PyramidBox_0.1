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
