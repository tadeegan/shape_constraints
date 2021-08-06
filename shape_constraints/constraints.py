
import inspect

RETURN_SIG = 'ret'

def _check_shape(param_name, shape, constraint):
  if len(shape) != len(constraint):
    raise ValueError(f'"{param_name}" has shape {shape} but constraint is {constraint}')
  template_values = {}
  for i, val in enumerate(constraint):
    if type(val) == str: # template type
      template_values[val] = shape[i]
    elif val != shape[i]:
      raise ValueError(f'"{param_name}" {shape} shape[{i}] does not match {val} in {constraint}')
  return template_values

def check_template_constraints(template_values):
  template_shapes = {}
  for val in template_values:
    for template, resolved_size in val.items():
      if template in template_shapes:
        if template_shapes[template] != resolved_size:
          raise ValueError(f'template <{template}> has mismatched values {template_shapes[template]} != {resolved_size}')
      else:
        template_shapes[template] = resolved_size

  
def shape_constraints(*args, **kwargs):
  constraints = kwargs
  def decorator(func):
    positional_parameter_names = list(inspect.signature(func).parameters.keys())
    for param_name in constraints.keys():
      if param_name == RETURN_SIG:
        continue
      if param_name not in positional_parameter_names:
        raise ValueError(f'"{param_name}" not found in function signature.')
    def wrapper(*args, **inner_kwargs):
      template_shapes = []
      for param_name, constraint in constraints.items():
        if param_name == RETURN_SIG:
          continue
        if param_name in inner_kwargs:
          template_shapes.append(_check_shape(param_name, inner_kwargs[param_name].shape, constraint))
          continue
        if param_name in positional_parameter_names:
          idx = positional_parameter_names.index(param_name)
          template_shapes.append(_check_shape(param_name, args[idx].shape, constraint))
          continue

      check_template_constraints(template_shapes)
      result = func(*args, **inner_kwargs)
      if RETURN_SIG in constraints:
        template_shapes.append(_check_shape('return value', result.shape, constraints[RETURN_SIG]))
      
      check_template_constraints(template_shapes)

      return result
    return wrapper
  return decorator