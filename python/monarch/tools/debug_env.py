# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os


_MONARCH_DEBUG_SERVER_HOST_ENV_VAR = "MONARCH_DEBUG_SERVER_HOST"
_MONARCH_DEBUG_SERVER_HOST_DEFAULT = "localhost"
_MONARCH_DEBUG_SERVER_PORT_ENV_VAR = "MONARCH_DEBUG_SERVER_PORT"
_MONARCH_DEBUG_SERVER_PORT_DEFAULT = "27000"
_MONARCH_DEBUG_SERVER_PROTOCOL_ENV_VAR = "MONARCH_DEBUG_SERVER_PROTOCOL"
_MONARCH_DEBUG_SERVER_PROTOCOL_DEFAULT = "tcp"


def _get_debug_server_host():
    return os.environ.get(
        _MONARCH_DEBUG_SERVER_HOST_ENV_VAR, _MONARCH_DEBUG_SERVER_HOST_DEFAULT
    )


def _get_debug_server_port():
    return os.environ.get(
        _MONARCH_DEBUG_SERVER_PORT_ENV_VAR, _MONARCH_DEBUG_SERVER_PORT_DEFAULT
    )


def _get_debug_server_protocol():
    return os.environ.get(
        _MONARCH_DEBUG_SERVER_PROTOCOL_ENV_VAR, _MONARCH_DEBUG_SERVER_PROTOCOL_DEFAULT
    )
