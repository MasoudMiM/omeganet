# Changelog

## 0.1.0 — first public release

Initial release of omagent, an LLM-assisted modeling toolkit for
OpenModelica, developed test-first and validated end-to-end against a live
omc (1.27.0) with a frontier LLM.

- `omagent.errors` — structured parsing of omc diagnostics
  (`getErrorString()`, simulation logs, and OMPython's exception format)
  with severity, source locations, and failure-kind classification
- `omagent.session` — `OMSession` wrapper over OMPython supporting both
  OMPython error contracts (return-and-drain, and raise-on-error), with
  honest per-operation success verdicts
- `omagent.loop` — `AgentLoop`: generate -> load -> check -> simulate ->
  verify with retry budget, LLM-backend-agnostic protocol, and
  environment-grounded lookup suggestions (queries omc `getClassNames` on
  "class not found" and injects near-miss hints into fix prompts)
- `omagent.results` — CSV and Dymola-format `.mat` result reading;
  quantitative verifiers (`expect_final`, `expect_value_at`,
  `expect_bounds` with settling windows, `expect_dips_below`, `all_of`)
  whose complaints feed back into the fix loop
- `omagent.llm` — Anthropic adapter with full round transcripts
- `omagent.tasks` / `omagent.runner` — a 5-tier benchmark ladder
  (pure equations -> MSL components -> hybrid events -> verifier-driven
  design -> multi-domain) with auto-gradable criteria and per-task
  transcript persistence
- 83 unit tests runnable without OpenModelica; 4 integration tests
  against a live omc
