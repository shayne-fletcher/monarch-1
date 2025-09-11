# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from unittest import TestCase

from monarch._rust_bindings.monarch_hyperactor.value_mesh import ValueMesh


class TestValueMesh(TestCase):
    def test_value_mesh(self) -> None:
        self.assertTrue(True)
