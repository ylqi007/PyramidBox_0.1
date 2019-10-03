# Copyright 2015 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Provides data for the Pascal VOC Dataset (images + annotations).
"""
import tensorflow as tf
from datasets import pascalvoc_common

slim = tf.contrib.slim

FILE_PATTERN = 'voc_2007_%s_*.tfrecord'


# Create a dictionary describing the features.
ITEMS_TO_DESCRIPTIONS = {
    'image/height': tf.io.FixedLenFeature([1], tf.int64),
    'image/width': tf.io.FixedLenFeature([1], tf.int64),
    'image/channels': tf.io.FixedLenFeature([1], tf.int64),
    'image/shape': tf.io.FixedLenFeature([3], tf.int64),
    'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/label': tf.io.VarLenFeature(dtype=tf.int64),
    'image/object/bbox/label_text': tf.io.VarLenFeature(dtype=tf.string),
    'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/raw_data': tf.io.FixedLenFeature((), tf.string, default_value=''),
}

# (Images, Objects) statistics on every class.
TRAIN_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (238, 306),
    'bicycle': (243, 353),
    'bird': (330, 486),
    'boat': (181, 290),
    'bottle': (244, 505),
    'bus': (186, 229),
    'car': (713, 1250),
    'cat': (337, 376),
    'chair': (445, 798),
    'cow': (141, 259),
    'diningtable': (200, 215),
    'dog': (421, 510),
    'horse': (287, 362),
    'motorbike': (245, 339),
    'person': (2008, 4690),
    'pottedplant': (245, 514),
    'sheep': (96, 257),
    'sofa': (229, 248),
    'train': (261, 297),
    'tvmonitor': (256, 324),
    'total': (5011, 12608),
}

TEST_STATISTICS = {
    'none': (0, 0),
    'aeroplane': (1, 1),
    'bicycle': (1, 1),
    'bird': (1, 1),
    'boat': (1, 1),
    'bottle': (1, 1),
    'bus': (1, 1),
    'car': (1, 1),
    'cat': (1, 1),
    'chair': (1, 1),
    'cow': (1, 1),
    'diningtable': (1, 1),
    'dog': (1, 1),
    'horse': (1, 1),
    'motorbike': (1, 1),
    'person': (1, 1),
    'pottedplant': (1, 1),
    'sheep': (1, 1),
    'sofa': (1, 1),
    'train': (1, 1),
    'tvmonitor': (1, 1),
    'total': (20, 20),
}

SPLITS_TO_SIZES = {
    'train': 5011,
    'test': 4952,
}
SPLITS_TO_STATISTICS = {
    'train': TRAIN_STATISTICS,
    'test': TEST_STATISTICS,
}
NUM_CLASSES = 20


def get_split(split_name, dataset_dir, file_pattern=None, items_to_description=None):
    """Gets a dataset tuple with instructions for reading ImageNet.
    Args:
        split_name: A train/test split name.
        dataset_dir: The base directory of the dataset sources.
        file_pattern: The file pattern to use when matching the dataset sources.
            It is assumed that the pattern contains a '%s' string so that the split
            name can be inserted.
        items_to_description: Description to parse tf.Example.
    Returns:
        A `Dataset` namedtuple.
    Raises:
        ValueError: if `split_name` is not a valid train/test split.
    """
    if not file_pattern:
        file_pattern = FILE_PATTERN
    return pascalvoc_common.get_dataset(split_name,
                                        dataset_dir,
                                        file_pattern,
                                        ITEMS_TO_DESCRIPTIONS)