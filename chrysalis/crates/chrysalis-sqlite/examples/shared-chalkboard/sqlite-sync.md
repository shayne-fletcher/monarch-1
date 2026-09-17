# Synchronize Two Chalkboards

This walkthrough starts after Alice has initialized `alice.db` and written `alice-1` as described in [`README.md`](README.md). Bob still has no database.

## Create Four Panes

```sh
tmux new-session -d -s shared-chalkboard -n demo \
  -e "chalkboard_demo_dir=$chalkboard_demo_dir" \
  -c "$HOME/fbsource/fbcode/monarch/chrysalis/crates/chrysalis-sqlite/examples/shared-chalkboard"
tmux split-window -h -t shared-chalkboard:demo
tmux split-window -v -t shared-chalkboard:demo.0
tmux split-window -v -t shared-chalkboard:demo.1
tmux select-layout -t shared-chalkboard:demo tiled
tmux select-pane -t shared-chalkboard:demo.0 -T 'Alice sync'
tmux select-pane -t shared-chalkboard:demo.1 -T 'Bob sync'
tmux select-pane -t shared-chalkboard:demo.2 -T 'Alice app'
tmux select-pane -t shared-chalkboard:demo.3 -T 'Bob app'
tmux select-pane -t shared-chalkboard:demo.0
tmux attach-session -t shared-chalkboard
```

The top panes hold the two long-running synchronizers. The bottom panes run short-lived commands against the corresponding local file.

## Start Alice

In the Alice sync pane, run:

```sh
cargo run -- sync "$chalkboard_demo_dir/alice.db"
```

The first output line is a join token such as:

```text
udp://127.0.0.1:43125?authority=0123456789abcdef0123456789abcdef
```

Copy the complete line. Alice is the root because she starts without a parent; this does not give her database authority over Bob's.

## Start Bob

In the Bob sync pane, substitute Alice's token and run:

```sh
alice_token='PASTE_ALICE_JOIN_TOKEN_HERE'
cargo run -- sync --join "$alice_token" "$chalkboard_demo_dir/bob.db"
```

Bob creates `bob.db`, installs the same trusted schema, connects to Alice, and prints his own join token. Leave both processes running.

## Observe Bidirectional Replication

In the Bob app pane:

```sh
cargo run -- show "$chalkboard_demo_dir/bob.db"
```

After replication catches up, Bob sees Alice's row:

```text
alice-1 | Alice | tea at five
```

Write Bob's row:

```sh
cargo run -- write "$chalkboard_demo_dir/bob.db" bob-1 Bob "bring biscuits"
```

In the Alice app pane, rerun `show` until both rows are visible:

```sh
cargo run -- show "$chalkboard_demo_dir/alice.db"
```

Then update Alice's existing row:

```sh
cargo run -- write "$chalkboard_demo_dir/alice.db" alice-1 Alice "tea at six"
```

Bob's next `show` eventually displays:

```text
alice-1 | Alice | tea at six
bob-1 | Bob | bring biscuits
```

Replication is asynchronous, so a `show` command may need to be repeated after a short pause.

## Stop the Synchronizers

Press Ctrl-C in Bob's sync pane, then in Alice's. Each process shuts down and joins its Chrysalis node before exiting. `show` still reads both database files locally after shutdown.

## How It Works

All four commands use the same trusted `TableSchema`. `init` and `sync` initialize that schema through `Replica::new`. `write` performs the SQL upsert inside `ReplicaTransaction`, captures the final row by its primary key, and commits the application row and replication log together.

The `sync` command creates an ephemeral development identity and loopback UDP carrier. It configures `ReplicationTopology` before constructing the Chrysalis `Node`. When Bob joins Alice, the node starts the typed SQLite protocol on their direct link.

The synchronizer and short-lived writer hold different SQLite connections in different processes. The replication loop therefore includes a periodic database-version check in addition to its in-process notification. That poll discovers a committed mutation from a short-lived writer and sends it to the peer.

Peers exchange trusted schema hashes, typed row changes, durable frontiers, and acknowledgements. They do not exchange executable schema SQL and do not infer mutations from arbitrary SQL.
