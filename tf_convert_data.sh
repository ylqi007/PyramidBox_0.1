#!/usr/bin/env bash

# In local
VOC2007_TRAIN="../../DATA/VOC2007/train/"
VOC2007_TEST="../../DATA/VOC2007/test/"

# In docker
#VOC2007_TRAIN="../VOC2007/train/"
#VOC2007_TEST="../VOC2007/test/"

DATA_OUTPUT="/tmp/pascalvoc_tfrecord/"

if [[ ! -d ${DATA_OUTPUT} ]]; then
  mkdir ${DATA_OUTPUT}
fi

# Convert Pascal VOC train to tfrecord files.
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${VOC2007_TRAIN} \
    --output_name=voc_2007_train \
    --output_dir=${DATA_OUTPUT}

# Convert Pascal VOC train to tfrecord files.
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=${VOC2007_TEST} \
    --output_name=voc_2007_test \
    --output_dir=${DATA_OUTPUT}