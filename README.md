# Shape Constraints Decorator Library

Tired of guessing what the correct tensor or matrix shapes to a function call?
Sure, you can document shapes in pydoc, but those can bit rot easily over time if someone forgets to update them or could simply be wrong.

Annotate your shapes with this @shape_constraints library! This will not only document the expected shapes to a function call but also ensure that the documentation for these shapes always stays up to date as they are enforced at run time. At a glance, the user knows exactly what shapes to provide.

For example:

```python
import numpy as np

def affine_transform(
    points: np.ndarray,
    T_FrameA_to_FrameB: np.ndarray):
    homogeneous_pts = np.append(points, np.ones((len(points),1)), axis=-1)
    return np.matmul(T_FrameA_to_FrameB, homogeneous_pts.T).T[..., :3]

pts = np.ones((3, 1000))
affine_transform(pts, np.eye(4))

>> ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1001 is different from 4)
```

The caller made a mistake about the shape of inputs. If instead we annotate the function @shape_constraints, they are less likely to make the same mistake and get a friendlyier error message:

```python
from shape_constraints import shape_constraints

@shape_constraints(
    points=('N', 3),
    T_FrameA_to_FrameB=(4, 4),
    ret=('N', 3)
)
def affine_transform(points, T_FrameA_to_FrameB):
    homogeneous_pts = np.append(points, np.ones((len(points),1)), axis=-1)
    return np.matmul(T_FrameA_to_FrameB, homogeneous_pts.T).T[..., :3]

pts = np.ones((3, 1000))
affine_transform(pts, np.eye(4))

>> ValueError: "points" (3, 1000) shape[1] does not match 3 in ('N', 3)

pts = np.ones((1000, 3))
affine_transform(pts, np.eye(4)) # Yay, passes
```

Example with PyTorch:

```python
import torch
import torch.nn.functional as F
import torch.nn as nn

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

input = torch.randn(4, 28, 28, 5)
label = torch.empty(4, 28, 28, dtype=torch.long).random_(5)
mask = torch.randn(4, 28, 28)

image_seg_classification_loss_per_image(input, label, mask)
``` 

### Semantics

```python
@shape_constraints(
    a=('A', 64),
    b=(64, 'A'),
    c=('A', 64),
    ret=(3,)
)
def some_func(a, b, c):
    return tf.reduce_sum(a + b.T + c, axis=0)
```

Inside a shape constraint, integers are directly used to contrain the shape. When using a string like "A" in this example, a template is created instead.  The concrete value of the template at run time is enforced to be equal across all contraints.

The return value can also contrained using "ret" parameter.

### Installation

shape_constraints is hosted on PyPi and can be installed with `pip install shape_constraints`.

### Compatibility

- [x] Numpy (np.ndarray)
- [x] PyTorch (torch.Tensor)
- [ ] PyTorch @jit mode. Need to check.
- [x] Tensorflow in eager mode
- [x] Tensorflow Keras in eager mode
- [ ] Tensorflow @tf.function jit. Need to check.
- [ ] Tensorflow v1 style graph apis.

# Development

To get started, install dev dependencies with `poetry install`.

Tests, can be run with `poetry run pytest tests`.