#!/usr/bin/env -S grimaldi --kernel monarch_demo_local
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# FILE_UID: f083db77-6303-4fbf-a9ec-8be986f0926a
# NOTEBOOK_NUMBER: N6185324 (8543474375772761)

""":md
Monarch: A single controller for distributed PyTorch
----------------------------------------------------
"""

""":md
## Setup
1. Create a new directory on your devgpu where some necessary packages will live.
2. Run `/data/users/$USER/fbsource/fbcode/monarch/examples/demo/setup.sh /path/to/your/dir`.
3. Connect this notebook to your devgpu and select the `monarch_demo (local)` kernel.
4. You can run the notebook in Bento Web by right-clicking on the demo.py tab in VS Code and selecting "Share Notebook to Bento Web". Some steps may fail in VS Code (e.g., displaying images and rendering simulation results).

See https://fburl.com/anp/xlypcyya for the last known working fbsource revision for this notebook.
"""

""":py"""
import getpass
import os
import sys

import monarch
import torch

import torch.nn as nn
from monarch.actor import this_host
from torch.utils._pytree import tree_map

torch.set_default_device("cuda")


""":md
Meshes
------
All computation is done on a 'mesh' of devices.
Here we create a mesh composed of the machine running the notebook:
"""

""":py"""
mesh = this_host().spawn_procs({"gpu": 8})
print(mesh.to_table())

""":md
Without a mesh active, torch runs locally.
"""

""":py"""
torch.rand(3, 4)

""":md
Once active, torch runs on every device in the mesh.
"""

""":py"""
with mesh.activate():
    t = torch.rand(3, 4, device="cuda")

""":py"""
t

""":md
Inspect moves rank0's copy of t to the notebook for debugging.
"""

""":py"""
monarch.inspect(t)

""":py"""
monarch.show(t)

""":md
Providing coordinates lets us inspect other ranks copies.
"""

""":py"""
monarch.show(t, gpu=1)

""":md
Tensor Commands
---------------

Any command done on the controller, such as multiplying these tensors,
performs that action to all of the tensors in the collection.
"""

""":py"""
with mesh.activate():
    obj = t @ t.T
    monarch.show(obj)

""":md
If a command fails, the workers stay alive and can execute future
commands:
"""

""":py"""

try:
    with mesh.activate():
        # too big
        big_w = torch.rand(4, 1024 * 1024 * 1024 * 1024 * 8, device="cuda")
        v = t @ big_w
        monarch.show(v)

except Exception as e:
    import traceback

    traceback.print_exc()

del big_w
print("RECOVERED!")

""":md
Since monarch recovers from errors, you can search for
what works:
"""

""":py"""
N = 1
while True:
    try:
        with mesh.activate():
            batch = torch.rand(N, 1024 * 1024 * 1024, device="cuda")
        monarch.inspect(batch.sum())
        N = 2 * N
        print(f"at least 2**{N} elements work")
    except Exception:
        print(f"max is 2**{N} elements")
        break

""":md
Collectives
-----------
Each machine has its own copy of the tensor, similar to torch.distributed.

To compute across tensors in the mesh, we use special communication operators, analogous to collectives.
"""

""":py"""
with mesh.activate():
    a = torch.rand(3, 4, device="cuda")
    r = a.reduce("gpu", "sum")

""":py"""
monarch.show(a, gpu=0)  # try
monarch.show(a, gpu=1)  # try

monarch.show(r, gpu=0)  # try
monarch.show(r, gpu=1)  # try

""":md
MAST GPUs
---------

We can also connect to remote GPUs reserved from some scheduler
"""

""":py"""


# NYI: schedule public API based on config, just fake it locally

mast_mesh = this_host().spawn_procs({"host": 4, "gpu": 4})


""":py"""
print(mast_mesh.to_table())
with mast_mesh.activate():
    eg = torch.rand(3, 4, device="cuda")
    rgpu = eg.reduce("gpu", "sum")
    rhost = eg.reduce("host", "sum")

""":md
Device Mesh Dimensions
----------------------

Meshes can be renamed and reshaped to fit the parallelism desired.
"""

""":py"""
mesh_2d_parallel = mast_mesh.rename(host="dp", gpu="tp")
print(mesh_2d_parallel.to_table())

""":py"""
mesh_3d_parallel = mast_mesh.split(host=("dp", "pp"), gpu=("tp",), pp=2)
print(mesh_3d_parallel.to_table())

""":md
Pipelining
----------

Pipelining is accomplish by slicing the mesh, and copying tensors from
one mesh to another.
"""

""":py"""
pipeline_mesh = mast_mesh.rename(host="pp")
meshes = [pipeline_mesh.slice(pp=i) for i in range(pipeline_mesh.size("pp"))]

""":py"""
print(meshes[0].to_table())

""":md
Intitialize a model across multiple meshes
"""

""":py"""
layers_per_stage = 2
stages = []
for stage_mesh in meshes:
    with stage_mesh.activate():
        layers = []
        for _ in range(layers_per_stage):
            layers.extend([nn.Linear(4, 4), nn.ReLU()])
        stages.append(nn.Sequential(*layers))

""":py"""


def forward_pipeline(x):
    with torch.no_grad():
        for stage_mesh, stage in zip(meshes, stages):
            x = x.to_mesh(stage_mesh)
            with stage_mesh.activate():
                x = stage(x)
        return x


""":py"""
with meshes[0].activate():
    input = torch.rand(3, 4, device="cuda")

output = forward_pipeline(input)
monarch.show(output)
print(output.mesh.to_table())

""":md
DDP Example
-----------

The next sections will use an example of writing DDP to illustrate a
typical way to develop code in monarch.

Let's interleave the backward pass with the gradient reductions and
parameter updates.

We use monarch.grad_generator to incrementally run the backward pass.
It returns an iterator that computes the grad parameters one at a time.
"""

""":py"""


def train(model, input, target):
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        it = iter(zip(rparameters, grads))
        todo = next(it, None)
        while todo is not None:
            param, grad = todo
            grad.reduce_("dp", "sum")
            todo = next(it, None)
            param += 0.01 * grad


""":md
Simulation of DDP
-----------------

We can use a simulator to check for expected behavior of code before running it
for real.

It is another kind of mesh, which simulates rather than computes results for real.
"""

""":py"""


def simulate():
    simulator = monarch.Simulator(hosts=1, gpus=4, trace_mode="stream_only")
    mesh = simulator.mesh.rename(gpu="dp")
    with mesh.activate():
        from model import Net

        model = Net()

        train(model, torch.rand(3, 4), torch.full((3,), 1, dtype=torch.int64))

        simulator.display()


""":py"""
# Make sure pop-ups are enabled in your browser for internalfb.com
simulate()

""":md
Overlapping Comms/Compute
-------------------------
Commands on different devices run in parallel,
but by default commands on a single device run sequentially.

We introduce parallelism on a device via stream objects.
"""

""":py"""
main = monarch.get_active_stream()
comms = monarch.Stream("comms")

""":md
<img src="https://lookaside.internalfb.com/intern/pixelcloudnew/asset/?id=468246752388474" width=500>
"""

""":md
To use a tensor from one stream on another we borrow it. The borrow API ensures determinstic memory usage,
and eliminates the race conditions in the torch.cuda.stream API.

<img src="https://lookaside.internalfb.com/intern/pixelcloudnew/asset/?id=556746733733298" width=500>
"""

""":md
The DDP example again, but using multiple streams.
"""

""":py"""


def train(model, input, target):
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        # NEW: iter also produces the tensor borrowed
        # to the comm stream
        it = iter(
            (param, grad, *comms.borrow(grad, mutable=True))
            for param, grad in zip(rparameters, grads)
        )

        todo = next(it, None)
        while todo is not None:
            param, grad, comm_grad, borrow = todo
            # NEW: compute the reduce on the comm stream
            with comms.activate():
                comm_grad.reduce_("dp", "sum")
            borrow.drop()
            todo = next(it, None)
            param += 0.01 * grad


simulate()

""":md
The simulation result showed the results did not overlap much
due to wherethe borrow.drop was placed.

<img src="https://lookaside.internalfb.com/intern/pixelcloudnew/asset/?id=1282606659410255" width=500>
"""

""":md
The goal is to get overlap like so:

<img src="https://lookaside.internalfb.com/intern/pixelcloudnew/asset/?id=1110575440645591" width=500>

We can achieve this by ending the borrow after the grad step but before
we update the param.
"""

""":py"""


def train(model, input, target):
    loss = model(input, target)
    rparameters = list(reversed(list(model.parameters())))
    grads = monarch.grad_generator(loss, rparameters)
    with torch.no_grad():
        it = iter(
            (param, grad, *comms.borrow(grad, mutable=True))
            for param, grad in zip(rparameters, grads)
        )

        todo = next(it, None)
        while todo is not None:
            param, grad, comm_grad, borrow = todo
            with comms.activate():
                comm_grad.reduce_("dp", "sum")
            todo = next(it, None)
            # NEW: delay the borrow as late as possible
            borrow.drop()
            param += 0.01 * grad


simulate()

""":py"""
