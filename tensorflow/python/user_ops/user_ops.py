# Copyright 2015 Google Inc. All Rights Reserved.
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

"""All user ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import gen_user_ops
from tensorflow.python.ops.gen_user_ops import *
from tensorflow.python.ops import common_shapes
import tensorflow as tf

def my_fact():
  """Example of overriding the generated code for an Op."""
  return gen_user_ops._fact()
  
def bt_decode_csv(records, feaLenLst, name=None):
  r"""_________________

  For decoding the svm like formate which used by DSSM.

  Args:
    records: A `Tensor` of type `string`.
    num_itm: An `int` that is `>= 1`.
    feaLenLst: An list of feature length of each item.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (itm_fea_lst, itm_len_lst).
    itm_fea_lst: A list of `num_itm` `Tensor` objects of type `float32`.
    itm_len_lst: A list of `num_itm` `Tensor` objects of type `int64`.
  """
  num_itm = len(feaLenLst)+1
  if num_itm < 3 :
    raise ValueError("Num of items must at lease be 3.")
  itm_fea_lst = gen_user_ops.bt_decode_csv(records, num_itm, name)
#   (itm_fea_lst,itm_len_lst) = output
  
  for idx in range(len(itm_fea_lst)):
    #print("-len-->"+str(feaLenLst[idx-1]))
    if idx >= num_itm :
      raise ValueError("Output too much item.")
    else :
      if idx == 0 :
        itm_fea_lst[idx] = tf.reshape(itm_fea_lst[idx],[1]) 
      else :
        itm_fea_lst[idx] = tf.reshape(itm_fea_lst[idx],[feaLenLst[idx-1]])
  return itm_fea_lst
  

# ops.RegisterShape("BtDecodeCSV")(common_shapes.unchanged_shape)
# @ops.RegisterShape("BtDecodeCSV")(common_shapes.unchanged_shape)
# def _BtDecodeCSVShape(op):  # pylint: disable=invalid-name
#   """Shape function for the BtDecodeCSV op."""
#   input_shape = op.inputs[0].get_shape()
#   # Optionally check that all of other inputs are scalar or empty.
#   for default_input in op.inputs[1:]:
#     default_input_shape = default_input.get_shape().with_rank(1)
#     if default_input_shape[0] > 1:
#       raise ValueError(
#           "Shape of a default must be a length-0 or length-1 vector.")
#   return [input_shape] * len(op.outputs)
