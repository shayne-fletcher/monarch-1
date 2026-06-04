# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

"""
Execution Surface Verifier (operator convenience)
=================================================

Fetches ``/v1/{actor}`` for a given actor reference and pretty-prints
the ``execution`` block, the second plane to ``actor_status``.

Limitation: this verifier is NOT the parent of ``execution_workload``,
so it cannot drive the stdin/stdout handshake that holds invocations.
It is a manual smoke / operator-convenience tool: point it at an actor
that is currently holding work (driven separately) to see the live
``execution`` shape. The deterministic end-to-end proof lives in the
Rust ``execution`` mesh_admin_integration test, which owns the
handshake.

By default it only prints. Pass ``--expect-active`` to additionally
assert ``active_handler_count >= 1`` with a populated ``active_handlers``
row (use only when you know the actor is mid-hold).

Exit codes::

  0  OK (printed; assertion passed if requested)
  1  FAIL (--expect-active requested but no active handler observed)
  2  ERROR (network / TLS / unexpected payload structure)
"""

import argparse
import json
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_ERROR = 2

FETCH_TIMEOUT_SEC = 10.0


class VerifierError(Exception):
    """Raised by helpers; ``main()`` classifies into ``EXIT_ERROR``."""


def build_ssl_context(args: argparse.Namespace) -> Optional[ssl.SSLContext]:
    if not args.cacert:
        return None
    ctx = ssl.create_default_context(cafile=args.cacert)
    ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
    return ctx


def _enc(ref: str) -> str:
    # Mesh-admin refs contain '/' inside actor ids; default safe='/'
    # would leave them unescaped. Match verify_inbound_ordering.py.
    return urllib.parse.quote(ref, safe="")


def fetch_json(url: str, ctx: Optional[ssl.SSLContext]) -> dict:
    try:
        with urllib.request.urlopen(
            url, context=ctx, timeout=FETCH_TIMEOUT_SEC
        ) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise VerifierError(f"GET {url} failed: {e}") from e
    except json.JSONDecodeError as e:
        raise VerifierError(f"GET {url} returned invalid JSON: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--admin-url", required=True)
    parser.add_argument("--actor", required=True, help="actor reference (unencoded)")
    parser.add_argument(
        "--expect-active",
        action="store_true",
        help="assert active_handler_count >= 1 with a populated row",
    )
    parser.add_argument("--cacert", default=None)
    parser.add_argument("--cert", default=None)
    parser.add_argument("--key", default=None)
    args = parser.parse_args()

    try:
        ctx = build_ssl_context(args)
        actor = fetch_json(f"{args.admin_url}/v1/{_enc(args.actor)}", ctx)
    except VerifierError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    actor_variant = actor.get("properties", {}).get("Actor")
    if actor_variant is None:
        print(
            f"ERROR: reference is not an Actor: {actor.get('properties')}",
            file=sys.stderr,
        )
        sys.exit(EXIT_ERROR)

    execution = actor_variant.get("execution")
    if execution is None:
        print("ERROR: Actor payload has no execution block", file=sys.stderr)
        sys.exit(EXIT_ERROR)

    print(json.dumps(execution, indent=2))

    if args.expect_active:
        count = execution.get("active_handler_count", 0)
        handlers = execution.get("active_handlers", [])
        if count < 1 or not handlers:
            print(
                f"FAIL: expected active handler; got active_handler_count={count}, "
                f"active_handlers={handlers}",
                file=sys.stderr,
            )
            sys.exit(EXIT_FAIL)
        print("OK: actor has at least one active handler")


if __name__ == "__main__":
    main()
