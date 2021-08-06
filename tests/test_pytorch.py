import torch
import torch.nn.functional as F
import torch.nn as nn
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

@shape_constraints(
    softmax_logits=('Batch', 'H', 'W', 'num_class'),
    labels=('Batch', 'H', 'W'),
    mask=('Batch', 'H', 'W'),
    ret=('Batch',)
)
def image_seg_classification_loss_per_image(softmax_logits, labels, mask):
  softmax_logits = softmax_logits.transpose(1, 3)
  loss = nn.CrossEntropyLoss(reduction='none')
  logits_flat = softmax_logits.flatten(start_dim=2)
  labels_flat = labels.flatten(start_dim=1)
  mask_flat = mask.flatten(start_dim=1)
  out = loss(logits_flat, labels_flat) * mask_flat
  return torch.sum(out, dim=1)

def test_torch_complicated():
  input = torch.randn(4, 28, 28, 5)
  label = torch.empty(4, 28, 28, dtype=torch.long).random_(5)
  mask = torch.randn(4, 28, 28)

  image_seg_classification_loss_per_image(input, label, mask)