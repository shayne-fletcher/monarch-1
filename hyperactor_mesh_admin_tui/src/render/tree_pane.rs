/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;
use ratatui::widgets::List;
use ratatui::widgets::ListItem;
use ratatui::widgets::ListState;

use crate::App;
use crate::VisibleRows;

const UNSELECTED_MARKER: &str = "  ";
const PANE_BORDER_WIDTH: usize = 2;

fn topology_row_text(rows: &VisibleRows<'_>, vis_idx: usize) -> String {
    let row = &rows.as_slice()[vis_idx];
    let node = row.node;
    let indent = "  ".repeat(row.depth);
    let connector = if row.depth == 0 {
        ""
    } else if rows.has_sibling_after(vis_idx, row.depth) {
        "├─ "
    } else {
        "└─ "
    };
    let fold = if node.has_children {
        if node.expanded { "▼ " } else { "▶ " }
    } else {
        "  "
    };

    format!("{}{}{}{}", indent, connector, fold, node.label)
}

/// Return the topology pane width, capped by its former percentage allocation.
///
/// TUI-28: sizing excludes cursor and scroll state. The marker allowance is
/// stable even if a future theme's selection caret is not two cells wide.
pub(crate) fn topology_pane_width(
    rows: &VisibleRows<'_>,
    pane_title: &str,
    selection_caret: &str,
    ceiling: u16,
) -> u16 {
    let marker_width = Line::from(selection_caret)
        .width()
        .max(Line::from(UNSELECTED_MARKER).width());
    let content_width = rows
        .as_slice()
        .iter()
        .enumerate()
        .map(|(vis_idx, _)| marker_width + Line::from(topology_row_text(rows, vis_idx)).width())
        .chain(std::iter::once(Line::from(pane_title).width()))
        .max()
        .unwrap_or_default()
        .saturating_add(PANE_BORDER_WIDTH);

    u16::try_from(content_width)
        .unwrap_or(u16::MAX)
        .min(ceiling)
}

/// Render the topology tree (left pane).
///
/// Uses `visible_rows()` to display only expanded nodes. Each row
/// includes indentation/connectors, an expand/collapse glyph for
/// nodes with children, and color-coding by `NodeType`, with the
/// selected row highlighted.
pub(crate) fn render_topology_tree(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    rows: &VisibleRows<'_>,
) {
    let scheme = &app.theme.scheme;

    // When any overlay is active the tree is non-interactive.
    // Render it uniformly dim so the user can see it is inactive.
    // This covers both the diagnostics overlay and the py-spy overlay.
    let pane_inactive = app.overlay.is_some();

    let items: Vec<ListItem> = rows
        .as_slice()
        .iter()
        .enumerate()
        .map(|(vis_idx, row)| {
            let node = row.node;

            // Style precedence: inactive > selected > failed > stopped >
            // system > node-type.  When pane_inactive the entire pane is dimmed
            // and the selection/failed/system colours must not bleed through.
            let style = if pane_inactive {
                scheme.detail_stopped
            } else if vis_idx == app.cursor.pos() {
                scheme.stat_selection.add_modifier(Modifier::BOLD)
            } else if node.failed {
                scheme.node_failed
            } else if node.stopped {
                scheme.detail_stopped
            } else if node.is_system {
                scheme.node_system_actor
            } else {
                scheme.node_user_actor
            };

            // Hide the selection caret while the pane is inactive.
            let marker = if !pane_inactive && vis_idx == app.cursor.pos() {
                app.theme.labels.selection_caret
            } else {
                UNSELECTED_MARKER
            };

            ListItem::new(Line::from(Span::styled(
                format!("{}{}", marker, topology_row_text(rows, vis_idx)),
                style,
            )))
        })
        .collect();

    let block = if pane_inactive {
        Block::default()
            .title(Span::styled(
                app.theme.labels.pane_topology,
                scheme.detail_stopped,
            ))
            .borders(Borders::ALL)
            .border_style(scheme.detail_stopped)
    } else {
        Block::default()
            .title(app.theme.labels.pane_topology)
            .borders(Borders::ALL)
            .border_style(scheme.border)
    };

    let list = List::new(items)
        .block(block)
        .highlight_style(Style::default());
    let mut list_state = ListState::default()
        .with_selected(Some(app.cursor.pos()))
        .with_offset(app.tree_scroll_offset);
    frame.render_stateful_widget(list, area, &mut list_state);
}

#[cfg(test)]
mod tests {
    use hyperactor_mesh::introspect::NodeRef;
    use ratatui::layout::Constraint;
    use ratatui::layout::Direction;
    use ratatui::layout::Layout;
    use ratatui::layout::Rect;

    use super::*;
    use crate::FlatRow;
    use crate::NodeType;
    use crate::TreeNode;

    fn node(label: &str, has_children: bool, expanded: bool) -> TreeNode {
        TreeNode {
            reference: NodeRef::Root,
            label: label.to_string(),
            node_type: NodeType::Root,
            expanded,
            fetched: true,
            has_children,
            stopped: false,
            failed: false,
            is_system: false,
            children: Vec::new(),
        }
    }

    fn rows<'a>(nodes: &'a [TreeNode], depths: &[usize]) -> VisibleRows<'a> {
        assert_eq!(nodes.len(), depths.len());
        VisibleRows::new(
            nodes
                .iter()
                .zip(depths)
                .map(|(node, &depth)| FlatRow {
                    node,
                    depth,
                    owning_proc: None,
                })
                .collect(),
        )
    }

    // TUI-28: short content shrinks to the title width plus borders.
    #[test]
    fn topology_width_hugs_short_content() {
        let nodes = [node("x", false, false)];
        let rows = rows(&nodes, &[0]);

        assert_eq!(topology_pane_width(&rows, "Topology", "▸ ", 40), 10);
    }

    // TUI-28: nested row width includes indentation, connector, fold, and borders.
    #[test]
    fn topology_width_includes_tree_structure() {
        let nodes = [node("p", true, true), node("child", false, false)];
        let rows = rows(&nodes, &[0, 1]);

        assert_eq!(topology_pane_width(&rows, "Topology", "▸ ", 40), 16);
    }

    // TUI-28: Unicode labels are measured in terminal cells, including when empty.
    #[test]
    fn topology_width_uses_unicode_display_cells() {
        let empty_rows = VisibleRows::new(Vec::new());
        assert_eq!(topology_pane_width(&empty_rows, "拓扑", "▸ ", 40), 6);

        let nodes = [node("节点", false, false)];
        let rows = rows(&nodes, &[0]);
        assert_eq!(topology_pane_width(&rows, "拓扑", "▸ ", 40), 10);
    }

    // TUI-28: the cap is Ratatui's exact former allocation, including rounding.
    #[test]
    fn topology_width_uses_former_percentage_allocation_as_ceiling() {
        let area = Rect::new(0, 0, 102, 10);
        let ceiling = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(area)[0]
            .width;
        assert_eq!(ceiling, 41);

        let nodes = [node(&"x".repeat(100), false, false)];
        let rows = rows(&nodes, &[0]);
        assert_eq!(topology_pane_width(&rows, "Topology", "▸ ", ceiling), 41);
    }
}
