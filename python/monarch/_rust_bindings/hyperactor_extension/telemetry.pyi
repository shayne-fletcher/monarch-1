# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def forward_to_tracing(message: str, file: str, lineno: int, level: int) -> None:
    """
    Log a message with the given metadata.

    Args:
    - message (str): The log message.
    - file (str): The file where the log message originated.
    - lineno (int): The line number where the log message originated.
    - level (int): The log level (10 for debug, 20 for info, 30 for warn, 40 for error).
    """
    ...
