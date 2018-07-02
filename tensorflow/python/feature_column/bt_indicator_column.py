# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.feature_column.feature_column import _DenseColumn
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.util.tf_export import tf_export

class _IndicatorColumn(_DenseColumn, 
                       collections.namedtuple('_IndicatorColumn',
                                              ['categorical_column'])):
  """Represents a one-hot column for use in deep networks.

  Args:
    categorical_column: A `_CategoricalColumn` which is created by
      `categorical_column_with_*` function.
  """

  @property
  def name(self):
    return '{}_indicator'.format(self.categorical_column.name)

  def _transform_feature(self, inputs):
    """Returns dense `Tensor` representing feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.

    Returns:
      Transformed feature `Tensor`.

    Raises:
      ValueError: if input rank is not known at graph building time.
    """
    id_weight_pair = self.categorical_column._get_sparse_tensors(inputs)  # pylint: disable=protected-access
    id_tensor = id_weight_pair.id_tensor
    weight_tensor = id_weight_pair.weight_tensor

    # If the underlying column is weighted, return the input as a dense tensor.
    if weight_tensor is not None:
      weighted_column = sparse_ops.sparse_merge(
          sp_ids=id_tensor,
          sp_values=weight_tensor,
          vocab_size=int(self._variable_shape[-1]))
      # Remove (?, -1) index
      weighted_column = sparse_ops.sparse_slice(weighted_column, [0, 0],
                                                weighted_column.dense_shape)
      # refer to 'https://github.com/tensorflow/tensorflow/issues/19876' and 'https://github.com/tensorflow/tensorflow/pull/19882/files'
      return array_ops.scatter_nd(weighted_column.indices, weighted_column.values, weighted_column.dense_shape)

    dense_id_tensor = sparse_ops.sparse_tensor_to_dense(
        id_tensor, default_value=-1)

    # One hot must be float for tf.concat reasons since all other inputs to
    # input_layer are float32.
    one_hot_id_tensor = array_ops.one_hot(
        dense_id_tensor,
        depth=self._variable_shape[-1],
        on_value=1.0,
        off_value=0.0)

    # Reduce to get a multi-hot per example.
    return math_ops.reduce_sum(one_hot_id_tensor, axis=[-2])

  @property
  def _parse_example_spec(self):
    return self.categorical_column._parse_example_spec  # pylint: disable=protected-access

  @property
  def _variable_shape(self):
    """Returns a `TensorShape` representing the shape of the dense `Tensor`."""
    return tensor_shape.TensorShape([1, self.categorical_column._num_buckets])  # pylint: disable=protected-access

  def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
    """Returns dense `Tensor` representing feature.

    Args:
      inputs: A `_LazyBuilder` object to access inputs.
      weight_collections: Unused `weight_collections` since no variables are
        created in this function.
      trainable: Unused `trainable` bool since no variables are created in
        this function.

    Returns:
      Dense `Tensor` created within `_transform_feature`.
    """
    # Do nothing with weight_collections and trainable since no variables are
    # created in this function.
    del weight_collections
    del trainable
    # Feature has been already transformed. Return the intermediate
    # representation created by _transform_feature.
    return inputs.get(self)

@tf_export('feature_column.bt_indicator_column')
def bt_indicator_column(categorical_column):
  """Represents multi-hot representation of given categorical column.

  Used to wrap any `categorical_column_*` (e.g., to feed to DNN). Use
  `embedding_column` if the inputs are sparse.

  ```python
  name = indicator_column(categorical_column_with_vocabulary_list(
      'name', ['bob', 'george', 'wanda'])
  columns = [name, ...]
  features = tf.parse_example(..., features=make_parse_example_spec(columns))
  dense_tensor = input_layer(features, columns)

  dense_tensor == [[1, 0, 0]]  # If "name" bytes_list is ["bob"]
  dense_tensor == [[1, 0, 1]]  # If "name" bytes_list is ["bob", "wanda"]
  dense_tensor == [[2, 0, 0]]  # If "name" bytes_list is ["bob", "bob"]
  ```

  Args:
    categorical_column: A `_CategoricalColumn` which is created by
      `categorical_column_with_*` or `crossed_column` functions.

  Returns:
    An `_IndicatorColumn`.
  """
  return _IndicatorColumn(categorical_column)
