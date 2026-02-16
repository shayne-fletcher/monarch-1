# Admin TUI Tree Refactor Plan (Direct Phase 3 Replacement)

## Goal
Replace the flat `Vec<TreeNode>` representation with a proper tree type and a deterministic flattening pass. This short-circuits the planned Phase 3 flat-list helpers and aligns with a structural tree abstraction.

## Data Model

Introduce a real tree node with children:

```rust
#[derive(Debug, Clone)]
struct TreeNode {
    reference: String,
    label: String,
    node_type: NodeType,
    expanded: bool,
    fetched: bool,
    has_children: bool,
    children: Vec<TreeNode>,
}
```

Notes:
- `children` stores the actual tree structure.
- `expanded` lives on the node; `flatten_visible` uses it to decide recursion.
- `has_children` and `fetched` preserve placeholder semantics (lazy fetch).

## Core Operations

### Build tree from API
Replace `build_subtree` (flat push) with a structural builder:

```rust
async fn build_subtree_tree(...) -> Option<TreeNode>
```

Behavior mirrors existing logic:
- Fetch node payload (respect filters).
- Build `TreeNode` with `children` as:
  - placeholders for Proc/Actor children (lazy).
  - eagerly fetched children for Root/Host.
- Respect `expanded_keys` when populating `children`.

### Flatten for UI
Derive visible rows from the tree:

```rust
struct FlatRow<'a> {
    node: &'a TreeNode,
    depth: usize,
}

fn flatten_visible<'a>(node: &'a TreeNode, depth: usize, out: &mut Vec<FlatRow<'a>>) {
    out.push(FlatRow { node, depth });
    if node.expanded {
        for child in &node.children {
            flatten_visible(child, depth + 1, out);
        }
    }
}
```

This replaces `visible_indices()` and all depth-based pruning logic.

### Expand / Collapse
- Expand: find node by reference (or by visible row index), fetch children if needed, set `expanded = true`.
- Collapse: set `expanded = false` (no removal needed).

### Lookup
A simple recursive search:

```rust
fn find_node_mut<'a>(node: &'a mut TreeNode, reference: &str) -> Option<&'a mut TreeNode>
```

If performance becomes an issue, introduce a reference-index map later.

## Migration Steps

1. **Replace tree storage**
   - `tree: Vec<TreeNode>` → `tree: Option<TreeNode>`

2. **Rewrite builder**
   - `build_subtree` returns a `TreeNode` with `children`.

3. **Replace visible indices**
   - `visible_indices()` becomes `flatten_visible()`.
   - Cursor indexes into flattened rows (same semantics as today).

4. **Update expand / collapse**
   - Expand: locate node in tree and populate its `children`.
   - Collapse: `expanded = false` on the node.

5. **Update rendering**
   - Render from `Vec<FlatRow>` instead of `visible_indices()`.
   - Indentation uses `depth` from flattened rows.

6. **Update selection**
   - Selection/cursor works over flattened rows.
   - Restore selection by matching `reference` + `depth` from flattened rows.

## Testing

- Add unit tests for `flatten_visible`:
  - Collapsed nodes hide descendants.
  - Expanded nodes include descendants with correct depth.
- Keep cursor tests unchanged.
- Manual TUI test:
  - Expand/collapse behavior unchanged.
  - Selection restore after refresh works.

## Expected Impact

- Removes flat-list tree surgery and related depth bookkeeping.
- Simplifies expand/collapse logic.
- Aligns with a principled tree representation (term-like structure).
- Paves the way for future algebraic tree folds if desired.
