monarch
=======

.. currentmodule:: monarch

The main Monarch module provides access to all public APIs for distributed computation.

Core Classes
------------

.. autoclass:: DeviceMesh
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RemoteProcessGroup
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Future
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: RemoteException
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Shape
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: NDSlice
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Selection
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Tensor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: OpaqueRef
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Stream
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Pipe
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ProcessAllocator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: LocalAllocator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: SimAllocator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ActorFuture
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Simulator
   :members:
   :undoc-members:
   :show-inheritance:

Core Functions
--------------

.. autofunction:: remote

.. autofunction:: coalescing

.. autofunction:: get_active_mesh

.. autofunction:: no_mesh

.. autofunction:: to_mesh

.. autofunction:: slice_mesh

.. autofunction:: create_pipe

.. autofunction:: remote_generator

.. autofunction:: get_active_stream

.. autofunction:: reduce

.. autofunction:: reduce_

.. autofunction:: call_on_shard_and_fetch

.. autofunction:: fetch_shard

.. autofunction:: inspect

.. autofunction:: show

.. autofunction:: grad_function

.. autofunction:: grad_generator

Mesh Creation Functions
-----------------------

.. autofunction:: python_local_mesh

.. autofunction:: mast_mesh

.. autofunction:: mast_reserve

.. autofunction:: rust_backend_mesh

.. autofunction:: rust_backend_meshes

.. autofunction:: local_mesh

.. autofunction:: local_meshes

.. autofunction:: rust_mast_mesh

.. autofunction:: world_mesh

Utilities
---------

.. autodata:: timer

.. autodata:: SocketType

.. autofunction:: set_meta

.. autodata:: builtins

.. autodata:: function_resolvers

Actor System (monarch.actor)
----------------------------

.. currentmodule:: monarch.actor

The `monarch.actor` module provides the actor-based programming model for distributed computation.

Actor Classes
~~~~~~~~~~~~~

.. autoclass:: Actor
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Accumulator
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ActorError
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Future
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Point
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Channel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: Port
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: PortReceiver
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: ValueMesh
   :members:
   :undoc-members:
   :show-inheritance:

Mesh Classes
~~~~~~~~~~~~

.. autoclass:: ProcMesh
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: HostMesh
   :members:
   :undoc-members:
   :show-inheritance:

Actor Functions
~~~~~~~~~~~~~~~

.. autofunction:: current_actor_name

.. autofunction:: as_endpoint

.. autofunction:: current_rank

.. autofunction:: current_size

.. autofunction:: endpoint

.. autofunction:: send

.. autofunction:: context

.. autofunction:: local_proc_mesh

.. autofunction:: proc_mesh

.. autofunction:: sim_proc_mesh

.. autofunction:: get_or_spawn_controller

.. autofunction:: this_host

.. autofunction:: this_proc

.. autofunction:: hosts_from_config

.. autofunction:: debug_controller
