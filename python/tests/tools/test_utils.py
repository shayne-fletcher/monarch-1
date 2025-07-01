# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import os
import unittest
from unittest import mock

from monarch.tools.utils import conda


class TestCondaUtils(unittest.TestCase):
    def test_conda_active_env_name(self) -> None:
        with mock.patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "foo-py3"}, clear=True):
            self.assertEqual(conda.active_env_name(), "foo-py3")

        with mock.patch.dict(
            os.environ, {"CONDA_PREFIX": "/home/USER/.conda/envs/bar-py3"}, clear=True
        ):
            self.assertEqual(conda.active_env_name(), "bar-py3")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(conda.active_env_name())

    def test_conda_active_env_dir(self) -> None:
        with mock.patch.dict(
            os.environ, {"CONDA_PREFIX": "/home/USER/.conda/envs/foo"}, clear=True
        ):
            self.assertEqual(conda.active_env_dir(), "/home/USER/.conda/envs/foo")

        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(conda.active_env_dir())
