from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

def count_split_examples(split_path, file_prefix='.tfrecord'):
    # Count the total number of examples in all of these shard
    num_samples = 0
    tfrecords_to_count = tf.gfile.Glob(os.path.join(split_path, file_prefix))
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):#, options = opts):
            num_samples += 1
    return num_samples

if __name__ == '__main__':
    print('train:', count_split_examples('/media/rs/7A0EE8880EE83EAF/Detections/SSD/dataset/tfrecords', 'train-?????-of-?????'))
    print('val:', count_split_examples('/media/rs/7A0EE8880EE83EAF/Detections/SSD/dataset/tfrecords', 'val-?????-of-?????'))
