from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import six

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.framework import dtypes

class LookupTableInterface(object):

  def __init__(self, key_dtype, value_dtype, name):
    """Construct a lookup table interface.

    Args:
      key_dtype: The table key type.
      value_dtype: The table value type.
      name: A name for the operation (optional).
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)
    self._name = name

  @property
  def key_dtype(self):
    """The table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    """The table value dtype."""
    return self._value_dtype

  @property
  def name(self):
    """The name of the table."""
    return self._name

  def size(self, name=None):
    """Compute the number of elements in this table."""
    raise NotImplementedError

  def lookup(self, keys, name=None):
    """Looks up `keys` in a table, outputs the corresponding values."""
    raise NotImplementedError


class BtIdTable(LookupTableInterface):
  """A generic mutable hash table implementation.

  Data can be inserted by calling the insert method. It does not support
  initialization via the init method.

  Example usage:

  ```python
  table = tf.contrib.lookup.BtIdTable(key_dtype=tf.string,
                                             value_dtype=tf.int64,
                                             default_value=-1)
  sess.run(table.insert(keys, values))
  out = table.lookup(query_keys)
  print(out.eval())
  ```
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               shared_name=None,
               name="BtIdTable",
               checkpoint=True):
    """

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      shared_name: If non-empty, this table will be shared under
        the given name across multiple sessions.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.

    Returns:
      A `BtIdTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(default_value,
                                                dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()

    # The table must be shared if checkpointing is requested for multi-worker
    # training to work correctly. Use the node name if no shared_name has been
    # explicitly specified.
    use_node_name_sharing = checkpoint and shared_name is None
    if self._default_value.get_shape().ndims == 0:
      self._table_ref = gen_lookup_ops.bt_id_table_v2(
          shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          name=name)
    else:
      self._table_ref = gen_lookup_ops.bt_id_table_of_tensors_v2(
          shared_name=shared_name,
          use_node_name_sharing=use_node_name_sharing,
          key_dtype=key_dtype,
          value_dtype=value_dtype,
          value_shape=self._default_value.get_shape(),
          name=name)
    super(BtIdTable, self).__init__(key_dtype, value_dtype,
                                           self._table_ref.op.name.split(
                                               "/")[-1])

    if checkpoint:
      saveable = BtIdTable._Saveable(self, name)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def size(self, name=None):
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self._name,
                        [self._table_ref]) as name:
      with ops.colocate_with(self._table_ref):
        return gen_lookup_ops.lookup_table_size_v2(self._table_ref, name=name)

  def lookup(self, keys, name=None):
    """

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(name, "%s_lookup_table_find" % self._name,
                        (self._table_ref, keys, self._default_value)) as name:
      with ops.colocate_with(self._table_ref):
        values = gen_lookup_ops.lookup_table_find_v2(
            self._table_ref, keys, self._default_value, name=name)

        values.set_shape(keys.get_shape().concatenate(self._value_shape))
    return values

  def export(self, name=None):
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self._name,
                        [self._table_ref]) as name:
      with ops.colocate_with(self._table_ref):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self._table_ref, self._key_dtype, self._value_dtype, name=name)

    exported_values.set_shape(exported_keys.get_shape().concatenate(
        self._value_shape))
    return exported_keys, exported_values

  class _Saveable(BaseSaverBuilder.SaveableObject):
    """SaveableObject implementation for BtIdTable."""

    def __init__(self, table, name):
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      # pylint: disable=protected-access
      super(BtIdTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, unused_restored_shapes):
      # pylint: disable=protected-access
      with ops.colocate_with(self.op._table_ref):
        return gen_lookup_ops.lookup_table_import_v2(
            self.op._table_ref, restored_tensors[0], restored_tensors[1])

class String2BtIdHelper(object):
  def __init__(self, shared_name=None):
    self._table = BtIdTable(
        key_dtype=dtypes.string,
        value_dtype=dtypes.int64,
        default_value=-1,
        shared_name=shared_name,
        name="BtIdTable_%s" % (shared_name) )
  
  def string_to_bt_id(self, input, name=None):
    value = self._table.lookup(keys=input, name=name)
    return value