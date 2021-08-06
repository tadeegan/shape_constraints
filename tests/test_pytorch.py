import torch
import pytest

from shape_constraints import shape_constraints

@shape_constraints(
    a=(3, 1),
    b=(3, 1),
    ret=(3, 3)
)
def sum(a: torch.Tensor, b: torch.Tensor):
  return torch.matmul(a, b.T)

def test_passing():
  a = torch.Tensor([[1],[2],[3]])
  b = torch.Tensor([[1],[2],[3]])
  sum(a, b) # positional
  sum(a=a, b=b) # named
  sum(a, b=b) # partial
