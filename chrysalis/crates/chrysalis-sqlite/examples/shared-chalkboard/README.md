# Shared Chalkboard

This example shows two independently writable SQLite files converging through two real Chrysalis processes.

Alice creates a chalkboard and writes the first note. Bob starts without a database. Each side then runs a long-lived `shared-chalkboard sync` process. Alice prints a join token, Bob connects with that token, and the typed `chrysalis-sqlite` protocol carries recorded rows in both directions.

The `init`, `write`, and `show` commands remain short-lived and local. They open only the selected database file. `write` records its mutation through a typed `ReplicaTransaction`; the synchronizer discovers that durable change and sends it on the next replication poll.

## Process Model

```text
short-lived Alice commands
            |
        alice.db
            |
long-lived Alice sync process
            |
      Chrysalis link
            |
long-lived Bob sync process
            |
         bob.db
            |
short-lived Bob commands
```

The synchronizers are separate operating-system processes with separate identities, runtimes, UDP sockets, and database files. Root and child describe the shape of their Chrysalis connection; neither database is primary.

## Commands

```text
shared-chalkboard init  <database>
shared-chalkboard write <database> <note-id> <author> <message>
shared-chalkboard show  <database>
shared-chalkboard sync  [--join <root-token>] <database>
```

`init` installs the trusted chalkboard schema and replication metadata. `write` performs a parameterized upsert, records the resulting row in the durable replication log, and exits. `show` reads local rows in stable note-ID order. `sync` attaches `ReplicationTopology` to a loopback Chrysalis node and remains alive until interrupted.

## Prepare Alice

```sh
cd ~/fbsource/fbcode/monarch/chrysalis/crates/chrysalis-sqlite/examples/shared-chalkboard
chalkboard_demo_dir=$(mktemp -d)

cargo run -- init "$chalkboard_demo_dir/alice.db"
cargo run -- write "$chalkboard_demo_dir/alice.db" alice-1 Alice "tea at four"
cargo run -- write "$chalkboard_demo_dir/alice.db" alice-1 Alice "tea at five"
cargo run -- show "$chalkboard_demo_dir/alice.db"
```

Alice now sees:

```text
alice-1 | Alice | tea at five
```

Nothing has created `bob.db`, and no Chrysalis node is running yet.

## Connect the Chalkboards

Continue with [`sqlite-sync.md`](sqlite-sync.md). It starts Alice and Bob in separate tmux panes, demonstrates replication in both directions, and explains how the typed replication path is assembled.

## What This Demonstrates

- separate processes can update separate local SQLite files;
- application writes explicitly record typed mutations rather than relying on SQL interception;
- a long-running replica detects durable writes made by another process;
- Chrysalis starts the SQLite protocol on a direct parent-child link; and
- stopping synchronization does not make either database dependent on the network for local reads.

## Non-goals

This example does not cover simultaneous edits to the same row, schema migration, reconnect behavior, multiple hosts, throughput, or a general-purpose SQL synchronization command.
