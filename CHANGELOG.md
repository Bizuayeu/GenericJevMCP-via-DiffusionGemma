# Changelog

## Unreleased

- Display per-question confidence and final-distribution entropy on separate lines before timing; expose the same values as structured metrics.

- Return typed total/decision timing and canonical display in MCP structuredContent/outputSchema; keep timing labels separate from question IDs.

- Group Python modules under jev/, operations under scripts/, and live probes under tests/live/. Use python -m jev.client and python -m jev.calibration from the repository root. MCP tool names and configuration/data locations are unchanged; upgrade the remote checkout together with the client.

## 0.1.0

Initial public release.

- DiffusionGemma NVFP4 structured decisions: yes/no, choice, zero-based score expectations.
- Optional text, source files and one PNG/JPEG image; explicit source-only grounding and abstention.
- Joint or sequential isolated questions, fixed/adaptive reads, candidate distributions and diagnostics.
- Standalone MCP tool, CLI JSON/text results and elapsed timing.
- Explicit local/SSH client configuration; no private corpus required for startup.
- Pinned model preparation, reproducible runtime overlay and offline calibration evaluator.
- English/Japanese setup, API semantics, measured latency boundaries and GB10 capacity limits.

初回公開。3型の判定・任意の文章／画像入力・MCP／CLI・実測秒数・資料なしの導入・固定runtime・オフライン校正評価を提供。日英READMEに導入条件と検証範囲を記載。
