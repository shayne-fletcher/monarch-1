# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import logging
import socket
from typing import Optional

logger: logging.Logger = logging.getLogger(__name__)


def get_ip_addr(hostname: str) -> str:
    """Resolves and returns the ip address of the given hostname.

    This function will return an ipv6 address if one that can bind
    `SOCK_STREAM` (TCP) socket is found. Otherwise it will fall-back
    to resolving an ipv4 `SOCK_STREAM` address.

    Raises a `RuntimeError` if neither ipv6 or ipv4 ip can be resolved from hostname.
    """

    def get_sockaddr(family: socket.AddressFamily) -> Optional[str]:
        try:
            # patternlint-disable-next-line python-dns-deps (only used for oss)
            addrs = socket.getaddrinfo(
                hostname, port=None, family=family, type=socket.SOCK_STREAM
            )  # tcp
            if addrs:
                # socket.getaddrinfo return a list of addr 5-tuple addr infos
                _, _, _, _, sockaddr = addrs[0]  # use the first address

                # sockaddr is a tuple (ipv4) or a 4-tuple (ipv6) where the first element is the ip addr
                ipaddr = str(sockaddr[0])

                logger.info(
                    "Resolved %s address: `%s` for host: `%s`",
                    family.name,
                    ipaddr,
                    hostname,
                )
                return str(ipaddr)
            else:
                return None
        except socket.gaierror as e:
            logger.info(
                "No %s address that can bind TCP sockets for host: %s. %s",
                family.name,
                hostname,
                e,
            )
            return None

    ipaddr = get_sockaddr(socket.AF_INET6) or get_sockaddr(socket.AF_INET)
    if not ipaddr:
        raise RuntimeError(
            f"Unable to resolve `{hostname}` to ipv6 or ipv4 address that can bind TCP socket."
            " Check the network configuration on the host."
        )
    return ipaddr
