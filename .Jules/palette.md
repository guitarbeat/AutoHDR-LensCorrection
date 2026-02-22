## 2026-02-22 - Reusable Drag & Drop Pattern
**Learning:** Simple `dragleave` causes flickering when dragging over children.
**Action:** Use `e.currentTarget.contains(e.relatedTarget)` to filter out events from children.
