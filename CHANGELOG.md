# Changelog

## 0.1.0 — 2026-11-19

### Added

1. Separate English and Simplified Chinese READMEs with installation, algorithm entries, configuration, architecture, session behavior, and validation scope.
2. MIT license, software citation metadata, contribution guide, issue and pull request templates, and a short Windows/Python 3.12 CI workflow.
3. Regression tests for all 18 algorithm entry points, learned-state persistence, DQN target updates, and UI session controls.
4. Light and dark themes, adjustable panels, keyboard shortcuts, and persistent preferences for text size, pacing, and TensorBoard opening.
5. A 32-second live application demonstration, current screenshots, and an isolated, repeatable recording script.

### Fixed

1. Preserve learned Q and V values during visualization synchronization and checkpoint restoration instead of replacing them with unused zero-initialized state.
2. Preserve actor policies when only critic parameters change.
3. Initialize DQN target parameters from the online network and carry the target-update counter across calls and checkpoint restoration.
4. Start or continue a prepared session without resetting it. Recreate logging when starting after Stop, and retain saved logs when closing the window.

### Clarified

1. The repository default is `QAC`; algorithm selection and environment rewards remain configurable in `main.py`.
2. The source display is a staged reading aid, and episode batches do not provide independent historical grid snapshots for every source assignment.
3. The deterministic actor-critic example uses discrete actions; the NumPy DQN is an educational implementation. Short regression checks do not establish convergence or benchmark performance.
