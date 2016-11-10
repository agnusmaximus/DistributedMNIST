from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.training import optimizer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
import sync_replicas_optimizer_modified.short_circuit_gradients as gradients

def get_valid_dtypes():
    """Valid types for loss, variables and gradients.
    Subclasses should override to allow other float types.
    Returns:
      Valid types for loss, variables and gradients.
    """
    return set([dtypes.float16, dtypes.float32, dtypes.float64])

def assert_valid_dtypes(tensors):
    """Asserts tensors are all valid types (see `_valid_dtypes`).
    Args:
      tensors: Tensors to check.
    Raises:
      ValueError: If any tensor is not a valid type.
    """
    valid_dtypes = get_valid_dtypes()
    for t in tensors:
      dtype = t.dtype.base_dtype
      if dtype not in valid_dtypes:
        raise ValueError(
            "Invalid type %r for %s, expected: %s." % (
                dtype, t.name, [v for v in valid_dtypes]))

def compute_gradients_with_injected_short_circuiting(loss, var_list=None,
                                                     gate_gradients=optimizer.Optimizer.GATE_OP,
                                                     aggregation_method=None,
                                                     colocate_gradients_with_ops=False,
                                                     sync_token_queue=None,
                                                     local_global_step=None,
                                                     global_step=None,
                                                     grad_loss=None):
    assert sync_token_queue is not None
    if gate_gradients not in [optimizer.Optimizer.GATE_NONE, optimizer.Optimizer.GATE_OP,
                              optimizer.Optimizer.GATE_GRAPH]:
        raise ValueError("gate_gradients must be one of: Optimizer.GATE_NONE, "
                         "Optimizer.GATE_OP, Optimizer.GATE_GRAPH.  Not %s" %
                         gate_gradients)
    assert_valid_dtypes([loss])
    if grad_loss is not None:
        assert_valid_dtypes([grad_loss])
    if var_list is None:
      var_list = variables.trainable_variables()
    for var in var_list:
      if not isinstance(var, variables.Variable):
        raise TypeError("Argument is not a tf.Variable: %s" % var)
    if not var_list:
      raise ValueError("No variables to optimize")
    var_refs = [v.ref() for v in var_list]
    grads = gradients.gradients_short_circuited(
        loss, var_refs, grad_ys=grad_loss,
        gate_gradients=(gate_gradients == optimizer.Optimizer.GATE_OP),
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops,
        sync_token_queue=sync_token_queue,
        local_global_step=local_global_step,
        global_step=global_step)
    if gate_gradients == optimizer.Optimizer.GATE_GRAPH:
        grads = control_flow_ops.tuple(grads)
    grads_and_vars = list(zip(grads, var_list))
    assert_valid_dtypes([v for g, v in grads_and_vars if g is not None])
    return grads_and_vars
