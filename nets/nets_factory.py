# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models.
"""
import tensorflow as tf

from nets import ssd_vgg_300

slim = tf.contrib.slim


networks_obj = {'ssd_300_vgg': ssd_vgg_300.SSDNet}


def get_network(name):
    """Get a network object from a name.
    """
    return networks_obj[name]
