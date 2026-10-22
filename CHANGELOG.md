# Changelog

Version history of AutoRefine, newest first. One line per round; the
per-round design detail lives in `SPEC.md` (the index in SPEC 0 maps every
milestone to its section and test). The version under development is
always the top row.

| Version | What changed | Detail |
| --- | --- | --- |
| 0.84.0 | Release gate (v0.84): packaging metadata + LICENSE, CLI help audit
  (five commands whose `--help` crashed are fixed), this changelog,
  `.gitattributes`, release-gate pins | SPEC 98 |
| v0.4 | Search Quality (v0.4 — implemented) | SPEC 18 |
| v0.6 | Environment & Loop Design (v0.6 — implemented) | SPEC 20 |
| v0.8 | Data-in CLI & one-page report (v0.8 — implemented) | SPEC 22 |
| v0.9 | Visual dashboard (v0.9 — implemented) | SPEC 23 |
| v0.10 | Input Modalities (v0.10 — implemented) | SPEC 24 |
| v0.11 | Modality-aware model families | SPEC 25 |
| v0.12 | Dashboard decision views | SPEC 26 |
| v0.13 | Dashboard acceptance-gate views | SPEC 27 |
| v0.14 | Dashboard learning views | SPEC 28 |
| v0.15 | Multi-run / policy views | SPEC 29 |
| v0.16 | Comprehension visuals II | SPEC 30 |
| v0.17 | Optimization | SPEC 31 |
| v0.18 | Optimization | SPEC 32 |
| v0.19 | Coherency | SPEC 33 |
| v0.20 | Coherency II | SPEC 34 |
| v0.21 | Coherency III | SPEC 35 |
| v0.22 | Generalization | SPEC 36 |
| v0.23 | Generalization: the canonical recipe & objective gates | SPEC 37 |
| v0.24 | Tracking: the run registry & lineage | SPEC 38 |
| v0.25 | Tracking II: live progress & decision accounting | SPEC 39 |
| v0.26 | Simulation: dry-run planning & counterfactual re-gating | SPEC 40 |
| v0.27 | Simulation III: projection, replay, and the narrated demo | SPEC 41 |
| v0.28 | v0.28 — Close the "use the model" loop (A32, M31) | SPEC 42 |
| v0.29 | v0.29 — Trust before you train (A33, M32) | SPEC 43 |
| v0.30 | v0.30 — Stable scores, visible features (A34, M33) | SPEC 44 |
| v0.31 | v0.31 — Adapt to more use cases (A35, M34) | SPEC 45 |
| v0.32 | v0.32 — Probabilistic outputs, curriculum beyond parity, portfolio mode (A36, M35) | SPEC 46 |
| v0.33 | Ergonomics | SPEC 47 |
| v0.34 | Beginner onboarding | SPEC 48 |
| v0.35 | Advanced analysis | SPEC 49 |
| v0.36 | Advanced: N-run comparison + one-click exports | SPEC 50 |
| v0.37 | App interactivity: tabs, stop/cancel, rejection reasons, accessibility | SPEC 51 |
| v0.38 | Interact with the run while it's alive | SPEC 52 |
| v0.39 | Show each decision, not just the running best | SPEC 53 |
| v0.40 | v0.40 — hover tooltips on the zero-hover views + the wall-time cost strip | SPEC 54 |
| v0.41 | Make it feel live | SPEC 55 |
| v0.42 | Professional polish | SPEC 56 |
| v0.43 | Research decision surfaces | SPEC 57 |
| v0.44 | The researcher round-2 surfaces | SPEC 58 |
| v0.45 | Parameters: visually interpreted and modified | SPEC 59 |
| v0.46 | Parameters: visually interpreted and modified — what-if & comparison | SPEC 60 |
| v0.47 | More specialized scenarios | SPEC 61 |
| v0.48 | Reporting for a wider range of audiences | SPEC 62 |
| v0.49 | Reporting for a wider range of audiences — formats, the user guide, the decision | SPEC 63 |
| v0.50 | Reporting for a wider range of audiences — the benchmark report + uncertainty on headlines | SPEC 64 |
| v0.51 | Quickstart in the UI | SPEC 65 |
| v0.52 | Decision robustness: does the verdict survive the seed band? | SPEC 66 |
| v0.53 | Self-improving RL — persistence, exploration, reuse | SPEC 67 |
| v0.54 | The RL loop in the dashboard | SPEC 68 |
| v0.55 | Workflow efficiency and result-view tabs | SPEC 69 |
| v0.56 | Hot-path execution performance | SPEC 70 |
| v0.57 | The visuals interface — cards, spacing, side-by-side layout | SPEC 71 |
| v0.58 | RL task-aware action masking — the learning-collapse fix | SPEC 72 |
| v0.59 | RL no-op step exclusion + REINFORCE sign correction — the collapse-root-cause fix | SPEC 73 |
| v0.60 | Training hot path, round 2 — loop invariants, in-place moments, in-place chains | SPEC 74 |
| v0.61 | Modeling capability v0.61 — `adamw` optimizer, three LR schedules, GP and GAM families | SPEC 75 |
| v0.62 | Workflow strip — finished is all-green | SPEC 76 |
| v0.63 | Workflow-strip settle — the strip shows the finished state | SPEC 77 |
| v0.64 | Modeling capability v0.64 — fine-pattern capture: Fourier features, GAM interactions, GP length scale | SPEC 78 |
| v0.65 | Data flow — the data-utilization diagram | SPEC 79 |
| v0.66 | Fine-tuning — start from a trained model | SPEC 80 |
| v0.67 | Model comparison — dynamic multi-model graphics | SPEC 81 |
| v0.68 | Nonlinearity intelligence — functionality and interconnection for nonlinear tasks | SPEC 82 |
| v0.69 | Native temporal models — sequence layout, conv1d / rnn families, the motif task | SPEC 83 |
| v0.70 | Sequential decision-making — sample weights, the cartpole sim, AWR, learned-model MPC | SPEC 84 |
| v0.71 | Sim-in-the-loop — the autonomous sequential improvement loop | SPEC 85 |
| v0.72 | Transfer to similar tasks — the search prior | SPEC 86 |
| v0.73 | Broader reach — zero-jargon in, plain understanding out, results in the reader's format | SPEC 87 |
| v0.74 | Minimum input, maximum return — found data, fair bars, reused runs, one-command hand-off | SPEC 88 |
| v0.75 | v0.75 — Time-based and on-demand data ingestion (A79) | SPEC 89 |
| v0.76 | v0.76 — On-demand UI: upload fresh data, adapt the model (A80) | SPEC 90 |
| v0.77 | v0.77 — Dense parallel + the optional GPU backend (A81) | SPEC 91 |
| v0.78 | v0.78 — Seven new data modalities + the upload→utilization flows (A82) | SPEC 92 |
| v0.79 | v0.79 — Extrapolation + intuitive transfer learning (A83) | SPEC 93 |
| v0.80 | v0.80 — Live steering: respond to changing parameters, and visualize the effects (A84) | SPEC 94 |
| v0.81 | v0.81 — The narrative: the environment's identity, in one screen (A85) | SPEC 95 |
| v0.82 | v0.82 — Pause & resume (A86) | SPEC 96 |
| v0.83 | v0.83 — Documentation agreement and succinctness (A87) | SPEC 97 |
