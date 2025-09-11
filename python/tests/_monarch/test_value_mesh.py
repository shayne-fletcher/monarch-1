# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

from monarch._rust_bindings.monarch_hyperactor.shape import Shape, Slice
from monarch._rust_bindings.monarch_hyperactor.value_mesh import ValueMesh


class TestValueMesh(TestCase):
    def test_construct(self) -> None:
        shape = Shape(["n"], Slice.new_row_major([3]))
        vm = ValueMesh(shape, [10, 20, 30])
        self.assertIsInstance(vm, ValueMesh)

    def test_len_and_values(self) -> None:
        shape = Shape(["n"], Slice.new_row_major([3]))
        vm = ValueMesh(shape, [10, 20, 30])
        self.assertEqual(len(vm), 3)
        self.assertEqual(vm.values(), [10, 20, 30])
