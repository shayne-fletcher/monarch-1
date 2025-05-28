# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, Optional

import torch

class GradientGenerator:
    def __init__(
        self,
        roots_list: Any,
        with_respect_to: Any,
        grad_roots: Any,
        context_restorer: Any,
    ): ...
    # pyre-ignore[11]: Annotation `torch.Tensor` is not defined as a type.
    def __next__(self) -> Optional[torch.Tensor]: ...
    def __iter__(self) -> "GradientGenerator": ...
