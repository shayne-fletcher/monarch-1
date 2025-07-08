# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

warnings.warn(
    "monarch.bootstrap_main is deprecated, please use from monarch._src.actor.bootstrap_main instead.",
    DeprecationWarning,
    stacklevel=2,
)

from monarch._src.actor.bootstrap_main import *  # noqa


if __name__ == "__main__":
    # noqa
    invoke_main()  # pragma: no cover
