monarch
=======

These API functions define monarch's distributed tensor computation API. See :doc:`../generated/examples/distributed_tensors` for an overview.

.. currentmodule:: monarch

.. autoclass:: Tensor
   :members:
   :show-inheritance:
   :exclude-members: __init__, get

.. autoclass:: Stream
   :members:
   :undoc-members:
   :show-inheritance:

.. autofunction:: remote

.. autofunction:: coalescing

.. autofunction:: get_active_mesh

.. autofunction:: no_mesh

.. autofunction:: to_mesh

.. autofunction:: slice_mesh

.. autofunction:: get_active_stream

.. autofunction:: reduce

.. autofunction:: reduce_

.. autofunction:: call_on_shard_and_fetch

.. autofunction:: fetch_shard

.. autofunction:: inspect

.. autofunction:: show

.. autofunction:: grad_function

.. autofunction:: grad_generator
