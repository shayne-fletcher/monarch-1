# Admin TUI Algebraic Refactoring Plan

**Goal**: Apply algebraic abstractions to reduce ~180-200 lines while improving elegance, minimalism, and generality.

**Principles**: Favor high-impact, code-reducing abstractions over structural reorganization.

---

## Phase 1: FetchState Semilattice (Highest Impact)

**Estimated reduction**: ~60-80 lines
**Files modified**: `hyperactor_mesh/bin/admin_tui.rs`

### Current Problems
1. Cache + detail + errors are three separate concerns (lines 180-194)
2. "Evict-then-fetch" hack to force refresh (lines 386-390)
3. Three duplicate fetch sites with inconsistent error handling:
   - `update_selected_detail` (lines 535-560)
   - `expand_node` (lines 419-433)
   - `build_subtree` (lines 746-749)

### Proposed Solution

**Add new type** (place near line 87, after `Args`):
```rust
use algebra::{JoinSemilattice, Max};
use hyperactor::clock::Clock;

/// Timestamped fetch state forming a semilattice where join prefers newer data.
#[derive(Clone, Debug)]
enum FetchState<T> {
    /// Not yet fetched or explicitly invalidated.
    Unknown,
    /// Successfully fetched at timestamp.
    Ready(Max<u64>, T),
    /// Failed to fetch at timestamp.
    Error(Max<u64>, String),
}

impl<T: Clone> JoinSemilattice for FetchState<T> {
    fn join(&self, other: &Self) -> Self {
        use FetchState::*;
        match (self, other) {
            (Unknown, x) | (x, Unknown) => x.clone(),
            (Ready(t1, v1), Ready(t2, _)) if t1 >= t2 => Ready(*t1, v1.clone()),
            (Ready(_, _), Ready(t2, v2)) => Ready(*t2, v2.clone()),
            (Error(t1, e1), Error(t2, _)) if t1 >= t2 => Error(*t1, e1.clone()),
            (Error(_, _), Error(t2, e2)) => Error(*t2, e2.clone()),
            (Ready(t1, v), Error(t2, _)) if t1 > t2 => Ready(*t1, v.clone()),
            (Error(t1, e), Ready(t2, _)) if t1 > t2 => Error(*t1, e.clone()),
            (x, _) => x.clone(),
        }
    }
}

impl<T> FetchState<T> {
    fn is_ready(&self) -> bool {
        matches!(self, FetchState::Ready(..))
    }

    fn as_ready(&self) -> Option<&T> {
        match self {
            FetchState::Ready(_, val) => Some(val),
            _ => None,
        }
    }

    fn as_error(&self) -> Option<&str> {
        match self {
            FetchState::Error(_, msg) => Some(msg),
            _ => None,
        }
    }
}
```

**Modify App struct** (lines 164-195):
```rust
struct App {
    base_url: String,
    client: reqwest::Client,
    should_quit: bool,

    tree: Vec<TreeNode>,
    selected: usize,

    // NEW: unified cache replaces node_cache + detail + detail_error + error
    cache: HashMap<String, FetchState<NodePayload>>,
    clock: RealClock,

    last_refresh: String,
    show_system_procs: bool,
}
```

**Add unified fetch method** (replace separate fetch logic):
```rust
impl App {
    /// Fetch or return cached node state. Caches the result (success or error).
    async fn fetch_or_cached(&mut self, reference: &str) -> FetchState<NodePayload> {
        let now = Max(self.clock.now().as_micros());

        match self.cache.get(reference) {
            Some(state @ FetchState::Ready(..)) => state.clone(),
            _ => {
                match fetch_node_raw(&self.client, &self.base_url, reference).await {
                    Ok(payload) => {
                        let state = FetchState::Ready(now, payload);
                        self.cache.insert(reference.to_string(), state.clone());
                        state
                    }
                    Err(e) => {
                        let state = FetchState::Error(now, e);
                        self.cache.insert(reference.to_string(), state.clone());
                        state
                    }
                }
            }
        }
    }

    /// Get the currently selected node's fetch state for the detail pane.
    fn selected_state(&self) -> Option<&FetchState<NodePayload>> {
        self.selected_tree_index()
            .and_then(|idx| self.tree.get(idx))
            .and_then(|node| self.cache.get(&node.reference))
    }
}
```

**Simplify `update_selected_detail`** (lines 535-560 → ~10 lines):
```rust
async fn update_selected_detail(&mut self) {
    if let Some(idx) = self.selected_tree_index() {
        if let Some(node) = self.tree.get(idx) {
            let reference = node.reference.clone();
            self.fetch_or_cached(&reference).await;
        }
    }
}
```

**Simplify `expand_node`** (lines 419-433 → ~8 lines):
```rust
// In expand_node, replace fetch logic:
let payload = match self.fetch_or_cached(&reference).await {
    FetchState::Ready(_, p) => p,
    FetchState::Error(_, e) => {
        // Error already cached, will show in detail pane
        return;
    }
    FetchState::Unknown => unreachable!(),
};
```

**Simplify `refresh`** (replace lines 386-390):
```rust
// Force refresh by invalidating cache for selected node
if let Some(idx) = self.selected_tree_index() {
    if let Some(node) = self.tree.get(idx) {
        self.cache.insert(node.reference.clone(), FetchState::Unknown);
    }
}
self.update_selected_detail().await;
```

**Update `render_detail_pane`** (lines 1338-1358):
```rust
fn render_detail_pane(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    match app.selected_state() {
        Some(FetchState::Ready(_, payload)) => render_node_detail(frame, area, payload),
        Some(FetchState::Error(_, msg)) => {
            let block = Block::default()
                .title("Details")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray));
            let p = Paragraph::new(Span::styled(msg, Style::default().fg(Color::Red)))
                .block(block);
            frame.render_widget(p, area);
        }
        _ => {
            let block = Block::default()
                .title("Details")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray));
            let p = Paragraph::new(Span::styled(
                "Select a node to view details",
                Style::default().fg(Color::Gray),
            ))
            .block(block);
            frame.render_widget(p, area);
        }
    }
}
```

**Update `render_header`** (lines 1210-1237):
```rust
fn render_header(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App) {
    // Check for root fetch error
    let status = if let Some(FetchState::Error(_, err)) = app.cache.get("root") {
        format!("ERROR: {}", err)
    } else {
        let sys = if app.show_system_procs {
            " | system procs: shown"
        } else {
            ""
        };
        format!(
            "Connected to {} | Last refresh: {}{}",
            app.base_url, app.last_refresh, sys
        )
    };
    // ... rest unchanged
}
```

**Update cache pruning in `refresh`** (lines 344-347):
```rust
// Prune stale cache entries
let live_refs: HashSet<&str> = self.tree.iter()
    .map(|n| n.reference.as_str())
    .collect();
self.cache.retain(|k, _| k == "root" || live_refs.contains(k.as_str()));
```

---

## Phase 2: Cursor Abstraction (Second Highest Impact)

**Estimated reduction**: ~30-40 lines
**Files modified**: `hyperactor_mesh/bin/admin_tui.rs`

### Current Problems
1. Selection bounds checking scattered across `on_key` (lines 582-612)
2. Manual clamping in multiple places (lines 377-380, 660-662)
3. No explicit "cursor laws" (e.g., invariant: `selected < visible.len()`)

### Proposed Solution

**Add Cursor type** (place after `FetchState`, near line 150):
```rust
/// Navigation cursor over a bounded list with movement operations.
///
/// Invariant: `pos < len` (unless `len == 0`, then `pos == 0`).
#[derive(Debug, Clone)]
struct Cursor {
    pos: usize,
    len: usize,
}

impl Cursor {
    fn new(len: usize) -> Self {
        Self { pos: 0, len }
    }

    /// Move up (decrement). Returns true if position changed.
    fn move_up(&mut self) -> bool {
        if self.pos > 0 {
            self.pos -= 1;
            true
        } else {
            false
        }
    }

    /// Move down (increment). Returns true if position changed.
    fn move_down(&mut self) -> bool {
        if self.pos + 1 < self.len {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// Jump to start. Returns true if position changed.
    fn home(&mut self) -> bool {
        if self.pos != 0 {
            self.pos = 0;
            true
        } else {
            false
        }
    }

    /// Jump to end. Returns true if position changed.
    fn end(&mut self) -> bool {
        let new_pos = self.len.saturating_sub(1);
        if self.pos != new_pos {
            self.pos = new_pos;
            true
        } else {
            false
        }
    }

    /// Update length and clamp position to remain valid.
    fn update_len(&mut self, new_len: usize) {
        self.len = new_len;
        if new_len == 0 {
            self.pos = 0;
        } else {
            self.pos = self.pos.min(new_len - 1);
        }
    }

    fn pos(&self) -> usize {
        self.pos
    }

    fn len(&self) -> usize {
        self.len
    }
}
```

**Modify App struct** (replace `selected: usize` at line 177):
```rust
struct App {
    // ...
    tree: Vec<TreeNode>,
    cursor: Cursor,  // Replaces: selected: usize
    // ...
}
```

**Update App::new** (line 229):
```rust
fn new(addr: &str) -> Self {
    Self {
        // ...
        tree: Vec::new(),
        cursor: Cursor::new(0),
        // ...
    }
}
```

**Simplify `on_key`** (lines 582-612 → ~20 lines):
```rust
fn on_key(&mut self, key: KeyEvent) -> KeyResult {
    match key.code {
        KeyCode::Char('q') | KeyCode::Esc => {
            self.should_quit = true;
            KeyResult::None
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            self.should_quit = true;
            KeyResult::None
        }
        KeyCode::Up | KeyCode::Char('k') => {
            if self.cursor.move_up() {
                KeyResult::DetailChanged
            } else {
                KeyResult::None
            }
        }
        KeyCode::Down | KeyCode::Char('j') => {
            if self.cursor.move_down() {
                KeyResult::DetailChanged
            } else {
                KeyResult::None
            }
        }
        KeyCode::Home | KeyCode::Char('g') => {
            if self.cursor.home() {
                KeyResult::DetailChanged
            } else {
                KeyResult::None
            }
        }
        KeyCode::End | KeyCode::Char('G') => {
            if self.cursor.end() {
                KeyResult::DetailChanged
            } else {
                KeyResult::None
            }
        }
        KeyCode::Tab => {
            if let Some(&tree_idx) = self.visible_indices().get(self.cursor.pos()) {
                if let Some(node) = self.tree.get_mut(tree_idx) {
                    if node.has_children && !node.expanded {
                        node.expanded = true;
                        return KeyResult::ExpandNode(tree_idx);
                    }
                }
            }
            KeyResult::None
        }
        // ... rest unchanged (collapse logic, etc.)
    }
}
```

**Update cursor length tracking**:

In `refresh` (after tree rebuild, line 342):
```rust
self.tree = tree;
let visible = self.visible_indices();
self.cursor.update_len(visible.len());
```

After collapse (line 661):
```rust
self.tree.retain(|n| n.depth == 0);
self.cursor.update_len(self.tree.len());
```

**Update all `self.selected` references**:
- Line 263: `visible.get(self.cursor.pos()).copied()`
- Line 1299: `if vis_idx == self.cursor.pos()`
- Line 1327: `.with_selected(Some(self.cursor.pos()))`

---

## Phase 3: Tree Operations (Factoring)

**Estimated reduction**: ~40-60 lines
**Files modified**: `hyperactor_mesh/bin/admin_tui.rs`

### Current Problems
1. Duplicate "remove children at depth" logic (lines 410-417, 637-644)
2. `visible_indices` is clear but could be a pure function
3. No reusable tree operations

### Proposed Solution

**Add tree operation helpers** (place near line 920, before `derive_label`):
```rust
// Tree operations on flat Vec<TreeNode> representation

/// Compute indices of visible nodes (those not hidden by collapsed ancestors).
fn visible_indices(tree: &[TreeNode]) -> Vec<usize> {
    let mut visible = Vec::new();
    let mut skip_below: Option<usize> = None;

    for (i, node) in tree.iter().enumerate() {
        if let Some(depth) = skip_below {
            if node.depth > depth {
                continue;
            }
            skip_below = None;
        }
        visible.push(i);
        if node.has_children && !node.expanded {
            skip_below = Some(node.depth);
        }
    }
    visible
}

/// Remove all descendants of the node at `idx` from the tree.
/// Returns the number of nodes removed.
fn remove_children_at(tree: &mut Vec<TreeNode>, idx: usize) -> usize {
    let depth = tree[idx].depth;
    let start = idx + 1;
    let end = tree.iter()
        .skip(start)
        .position(|n| n.depth <= depth)
        .map_or(tree.len(), |pos| start + pos);
    tree.drain(start..end);
    end - start
}

/// Insert children nodes immediately after the parent at `idx`.
/// First removes any existing children (for re-expansion).
fn insert_children_at(tree: &mut Vec<TreeNode>, idx: usize, children: Vec<TreeNode>) {
    remove_children_at(tree, idx);
    let insert_pos = idx + 1;
    tree.splice(insert_pos..insert_pos, children);
}
```

**Refactor App methods to use helpers**:

Update `visible_indices` method (lines 239-256 → 3 lines):
```rust
fn visible_indices(&self) -> Vec<usize> {
    visible_indices(&self.tree)
}
```

Update `expand_node` (lines 409-417 → 2 lines):
```rust
async fn expand_node(&mut self, tree_idx: usize) {
    let reference = self.tree[tree_idx].reference.clone();
    let depth = self.tree[tree_idx].depth;

    // Remove existing children (if any)
    remove_children_at(&mut self.tree, tree_idx);

    // ... rest of fetch logic unchanged ...

    // Insert children (line 524 changes):
    insert_children_at(&mut self.tree, tree_idx, child_nodes);
}
```

Update collapse logic in `on_key` (lines 631-644 → 8 lines):
```rust
KeyCode::BackTab => {
    if let Some(&tree_idx) = visible.get(self.cursor.pos()) {
        let should_collapse = self
            .tree
            .get(tree_idx)
            .map(|n| n.has_children && n.expanded)
            .unwrap_or(false);
        if should_collapse {
            self.tree[tree_idx].expanded = false;
            remove_children_at(&mut self.tree, tree_idx);
            return KeyResult::DetailChanged;
        }
    }
    KeyResult::None
}
```

---

## Phase 4: Renderable Refactor (Polish)

**Estimated reduction**: ~30-50 lines
**Files modified**: `hyperactor_mesh/bin/admin_tui.rs`

### Current Problems
1. Four separate `render_*_detail` functions with repetitive block creation (lines 1415-1659)
2. Dispatcher at lines 1369-1412 is verbose

### Proposed Solution

**Extract detail line builders** (replace individual render functions):
```rust
// Replace render_root_detail, render_host_detail, etc. with simple builders

fn build_root_lines(payload: &NodePayload, num_hosts: usize) -> Vec<Line> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Hosts: ", Style::default().fg(Color::Gray)),
            Span::raw(num_hosts.to_string()),
        ]),
        Line::default(),
    ];
    for child in &payload.children {
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(child, Style::default().fg(Color::Cyan)),
        ]));
    }
    lines
}

fn build_host_lines(payload: &NodePayload, addr: &str, num_procs: usize) -> Vec<Line> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Address: ", Style::default().fg(Color::Gray)),
            Span::raw(addr),
        ]),
        Line::from(vec![
            Span::styled("Procs: ", Style::default().fg(Color::Gray)),
            Span::raw(num_procs.to_string()),
        ]),
        Line::default(),
    ];
    for child in &payload.children {
        let short = ProcId::from_str(child)
            .ok()
            .and_then(|pid| pid.name().cloned())
            .unwrap_or_else(|| child.clone());
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::styled(short, Style::default().fg(Color::Green)),
        ]));
    }
    lines
}

fn build_proc_lines(payload: &NodePayload, proc_name: &str, num_actors: usize) -> Vec<Line> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled("Name: ", Style::default().fg(Color::Gray)),
            Span::raw(proc_name),
        ]),
        Line::from(vec![
            Span::styled("Actors: ", Style::default().fg(Color::Gray)),
            Span::raw(num_actors.to_string()),
        ]),
        Line::default(),
    ];
    for (i, actor) in payload.children.iter().enumerate() {
        if i >= 50 {
            lines.push(Line::from(Span::styled(
                format!("  ... and {} more", payload.children.len() - 50),
                Style::default().fg(Color::DarkGray),
            )));
            break;
        }
        lines.push(Line::from(vec![
            Span::styled("  ", Style::default()),
            Span::raw(actor),
        ]));
    }
    lines
}

// Keep build_actor_lines separate due to two-pane layout
```

**Simplify render_node_detail** (lines 1368-1413 → ~25 lines):
```rust
fn render_node_detail(frame: &mut ratatui::Frame<'_>, area: Rect, payload: &NodePayload) {
    match &payload.properties {
        NodeProperties::Root { num_hosts } => {
            let lines = build_root_lines(payload, *num_hosts);
            let block = Block::default()
                .title("Root Details")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray));
            let p = Paragraph::new(lines).block(block);
            frame.render_widget(p, area);
        }
        NodeProperties::Host { addr, num_procs } => {
            let lines = build_host_lines(payload, addr, *num_procs);
            let block = Block::default()
                .title("Host Details")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray));
            let p = Paragraph::new(lines).block(block);
            frame.render_widget(p, area);
        }
        NodeProperties::Proc { proc_name, num_actors, .. } => {
            let lines = build_proc_lines(payload, proc_name, *num_actors);
            let block = Block::default()
                .title("Proc Details")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Gray));
            let p = Paragraph::new(lines).block(block);
            frame.render_widget(p, area);
        }
        NodeProperties::Actor { .. } => {
            // Keep current two-pane actor rendering (too complex to factor)
            render_actor_detail(frame, area, payload, /* ... */);
        }
        NodeProperties::Error { code, message } => {
            let text = format!("Error: {} — {}", code, message);
            let paragraph = Paragraph::new(text)
                .block(Block::default().borders(Borders::ALL).title("Error"))
                .wrap(Wrap { trim: true });
            frame.render_widget(paragraph, area);
        }
    }
}
```

**Note**: Actor rendering (lines 1545-1659) stays mostly unchanged due to two-pane layout complexity. Main win is consolidating the simpler node types.

---

## Implementation Checklist

### Phase 1: FetchState Semilattice
- [ ] Add `FetchState<T>` enum and `JoinSemilattice` impl
- [ ] Add `use algebra::{JoinSemilattice, Max};` import
- [ ] Update `App` struct: replace `node_cache`, `detail`, `detail_error`, `error` with `cache`
- [ ] Add `clock: RealClock` to `App`
- [ ] Add `fetch_or_cached` method
- [ ] Add `selected_state` method
- [ ] Simplify `update_selected_detail`
- [ ] Simplify `expand_node` fetch logic
- [ ] Simplify `refresh` invalidation logic
- [ ] Update `render_detail_pane` to use `FetchState`
- [ ] Update `render_header` to use `FetchState`
- [ ] Update `build_subtree` to use new cache
- [ ] Run `cargo fmt && cargo clippy`
- [ ] Test: basic navigation, expansion, error handling

### Phase 2: Cursor Abstraction
- [ ] Add `Cursor` struct with movement methods
- [ ] Update `App` struct: replace `selected` with `cursor`
- [ ] Update `App::new` to initialize cursor
- [ ] Refactor `on_key` navigation logic
- [ ] Update cursor length in `refresh`
- [ ] Update cursor length after collapse
- [ ] Replace all `self.selected` with `self.cursor.pos()`
- [ ] Run `cargo fmt && cargo clippy`
- [ ] Test: all navigation keys (j/k, g/G, home/end)

### Phase 3: Tree Operations
- [ ] Add `visible_indices` free function
- [ ] Add `remove_children_at` function
- [ ] Add `insert_children_at` function
- [ ] Refactor `App::visible_indices` to call free function
- [ ] Refactor `expand_node` to use helpers
- [ ] Refactor collapse in `on_key` to use helpers
- [ ] Run `cargo fmt && cargo clippy`
- [ ] Test: expand/collapse, collapse-all

### Phase 4: Renderable Refactor
- [ ] Extract `build_root_lines`
- [ ] Extract `build_host_lines`
- [ ] Extract `build_proc_lines`
- [ ] Simplify `render_node_detail` dispatcher
- [ ] Remove old `render_root_detail`, `render_host_detail`, `render_proc_detail`
- [ ] Run `cargo fmt && cargo clippy`
- [ ] Test: all node type detail views

### Final
- [ ] Run full test suite
- [ ] Verify line count reduction (~180-200 lines)
- [ ] Update comments to reflect new abstractions
- [ ] Commit with message: "refactor(admin_tui): apply algebraic abstractions"

---

## Expected Outcomes

**Before**: ~1674 lines (admin_tui.rs)
**After**: ~1470-1490 lines
**Reduction**: ~180-200 lines (10-12%)

**Improvements**:
1. **Correctness**: Fetch state joins are timestamp-based and commutative
2. **Simplicity**: Three fetch sites → one unified path
3. **Clarity**: Cursor laws explicit, tree operations reusable
4. **Generality**: FetchState and Cursor are reusable beyond this TUI

**Trade-offs**:
- Slight increase in type system complexity (FetchState enum)
- Requires understanding of join-semilattice semantics
- Small runtime cost for timestamp tracking (negligible)
