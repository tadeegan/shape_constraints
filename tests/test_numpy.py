import numpy as np
import pytest

from shape_constraints import shape_constraints

@shape_constraints(
    a=(3, 1),
    b=(3, 1),
    ret=(3, 1)
)
def sum(a: np.ndarray, b: np.ndarray):
  return a + b

class MyClass():
  def __init__(self, const):
    self.const = const
  
  @shape_constraints(a=('N', 1), ret=('N', 1))
  def add(self, a):
    return a + self.const

def test_passing():
  a = np.array([[1],[2],[3]])
  b = np.array([[1],[2],[3]])
  sum(a, b) # positional
  sum(a=a, b=b) # named
  sum(a, b=b) # partial

  mc = MyClass(10)
  mc.add(a)

def test_bad_params():
  # Wrong shape size
  with pytest.raises(ValueError) as error:
    sum(np.array([1,2,3]), b=np.array([[1],[2],[3]]))
  assert '"a" has shape (3,) but constraint is' in str(error.value)

  # Wrong outer dim
  with pytest.raises(ValueError) as error:
    a = np.ones((4,1))
    b = np.ones((3,1))
    sum(a, b)
  assert 'shape[0] does not match 3' in str(error.value)
  
  # Wrong inner dim
  with pytest.raises(ValueError) as error:
    a = np.ones((3,2))
    b = np.ones((3,1))
    sum(a, b)
  assert 'shape[1] does not match 1' in str(error.value)

@shape_constraints(
    a=(3, 1),
    b=(3, 1),
    ret=(3, 1)
)
def sum_reshape(a: np.ndarray, b: np.ndarray):
  return np.squeeze(a + b)

def test_return_fails():
  a = np.array([[1],[2],[3]])
  b = np.array([[1],[2],[3]])
  with pytest.raises(ValueError) as error:
    sum_reshape(a, b) # positional

@shape_constraints(
    a=('N', 1),
    b=(3, 'N'),
    ret=(3, 'N')
)
def transpose_then_mul_templated(a: np.ndarray, b: np.ndarray):
  return a.T * b

def test_templated():
  a = np.ones((100, 1))
  b = np.ones((3, 100))
  transpose_then_mul_templated(a, b)

  a = np.ones((100, 1))
  b = np.ones((3, 1100))
  with pytest.raises(ValueError) as error:
    transpose_then_mul_templated(a, b)

def test_not_matching_parameter():
  with pytest.raises(ValueError) as error:
    @shape_constraints(
      c = (100, 2)
    )
    def a(a, b):
      return a + b
