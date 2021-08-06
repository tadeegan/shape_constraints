import tensorflow as tf
import pytest

from shape_constraints import shape_constraints

@shape_constraints(
    a=(3, 1),
    b=(3, 1),
    ret=(3, 3)
)
def sum(a: tf.Tensor, b: tf.Tensor):
  return tf.matmul(a, tf.transpose(b))

def test_passing():
  a = tf.constant([[1],[2],[3]])
  b = tf.constant([[1],[2],[3]])
  sum(a, b) # positional
  sum(a=a, b=b) # named
  sum(a, b=b) # partial
