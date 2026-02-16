# RFC: Admin TUI Help/Capabilities Pane (SKILL.md Rendering)

## Summary
Add a Help/Capabilities pane to the Admin TUI that fetches and displays the Mesh Admin `SKILL.md` document. The goal is to improve discoverability and operational clarity directly inside the TUI without introducing action/capability execution yet.

## Motivation
- The Mesh Admin HTTP server already exposes `/SKILL.md` with self‑describing API documentation.
- Operators and developers often need to recall endpoints, semantics, or troubleshooting hints while in the TUI.
- Surfacing this in‑context reduces context switches and accelerates onboarding.

## Goals
- Provide a toggleable help pane (e.g., `?` or `h`) in the Admin TUI.
- Fetch `/SKILL.md` once per session and cache it.
- Render the content in the TUI (initially plain text; optional markdown rendering later).

## Non‑Goals (for now)
- No action execution or capabilities from the TUI.
- No per‑agent SKILL documents yet (e.g., `/v1/{reference}/SKILL.md`).
- No live refresh polling of SKILL content.

## Proposed UX
- Add a hotkey to toggle a right‑hand overlay or split pane.
- When opened the first time, fetch `/SKILL.md` from the same base URL used for node queries.
- Display cached content in a scrollable text view.
- Show a small status line with base URL and fetch timestamp (optional).

## Technical Approach

### Data flow
- Add `help_visible: bool` and `help_text: Option<String>` to `App`.
- On toggle:
  - If `help_text` is `None`, fetch `GET {base_url}/SKILL.md`.
  - Cache the response in `help_text`.

### Rendering
- If `help_visible` is true, render a help pane on the right (or an overlay).
- Initially render as plain text using `Paragraph` + `Wrap`.
- Optionally add a scroll offset (`help_scroll`) if text exceeds viewport.

### Optional markdown rendering
- If desired later, add a markdown parser and map to ratatui spans.
- Candidate: `pulldown-cmark` with a small renderer that maps headings/bold/links to styles.
- Keep this optional to avoid dependency creep in the first iteration.

## Alternatives Considered
- External docs only (status quo): slower, context switching.
- Inline action palette without docs: premature given current scope.
- Per‑agent SKILL docs: promising but deferred until actions/capabilities are defined.

## Risks
- Minimal: only adds one HTTP fetch and cached text rendering.
- TUI real estate: ensure the help pane can be dismissed quickly and doesn’t block primary view.

## Open Questions
- Preferred hotkey (`?` vs `h`)?
h
- Pane placement: overlay vs split view?
let's start with overlay. if we don't like we can reconsider later.
- Should we show a one‑line hint in the footer (“? for help”) by default?
yes.

## Future Work
- Per‑agent `SKILL.md` at `/v1/{reference}/SKILL.md` authored by actor implementors.
- Action/capability execution once schemas are defined.
- Markdown rendering with richer styles and link hints.
i'm very interested in integrating a markdown crate now. i don't mind the dep.
