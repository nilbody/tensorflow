from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import six

from tensorflow.python.ops import btid_table_ops.py
from tensorflow.python.framework import constant_op
import tensorflow as tf

def test_string_to_id():
  key1 = constant_op.constant(["brain","salad","surgery"])
  key2 = constant_op.constant(["hello","hi"])
  key3 = constant_op.constant(["brain","hi"])

  helper1 = btid_table_ops.String2BtIdHelper("share_name")
  helper1_id1 = helper1.string_to_bt_id(key1)
  helper1_id2 = helper1.string_to_bt_id(key2)
  helper1_id3 = helper1.string_to_bt_id(key3)

  helper2 = btid_table_ops.String2BtIdHelper("share_name")
  helper2_id3 = helper2.string_to_bt_id(key3)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)

    helper1_result1 = sess.run(helper1_id1)
    helper1_result2 = sess.run(helper1_id2)
    helper1_result3 = sess.run(helper1_id3)
    helper2_result3 = sess.run(helper2_id3)

    print("helper1_result1, %d", helper1_result1)
    print("helper1_result2, %d", helper1_result2)
    print("helper1_result3, %d", helper1_result3)
    print("helper2_result3, %d", helper2_result3)

