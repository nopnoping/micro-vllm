# micro-vLLM Engineering Guidelines

## Scope and Priorities

This file contains stable, repository-wide rules. Keep version requirements,
implementation plans, and temporary task notes outside it.

Apply these priorities in order:

1. Correctness.
2. Clarity and educational value.
3. Stable, minimal abstractions.
4. Measured performance.
5. Feature coverage.

Prefer a small implementation with explicit, tested invariants over broader but
opaque behavior.

## Architecture Boundaries

- `engine` owns public orchestration and request lifecycle coordination.
- `core` owns device-independent scheduling and logical cache state.
- `executor` owns execution interfaces and worker lifecycle coordination.
- `worker` owns device resources and turns execution plans into model runs.
- `models` owns model assembly and architecture-specific behavior.
- `layers` owns reusable neural-network layers and execution backends.
- `sampling` owns sampling configuration and logits-to-token behavior.
- `weights` owns checkpoint discovery, mapping, and parameter loading.

Enforce these boundaries:

- Core code must not invoke model kernels or own device tensors.
- The scheduler decides which requests run or are preempted; the KV cache manager
  owns logical block allocation, sharing, and release mechanisms.
- Executors and model runners must not mutate scheduler-owned request state.
- Models and layers must not allocate, free, or preempt logical cache blocks.
- Workers own physical device tensors; the control plane owns logical IDs and
  lifecycle state.
- Shared schemas must not import concrete implementations or create dependency
  cycles.

## Design Rules

- Pass configuration and runtime context explicitly. Avoid mutable global state,
  hidden singletons, duplicate counters, and derived phase flags that can diverge.
- Rely on type annotations and static checking for parameter types. Reserve
  runtime validation for domain values, state transitions, and invariants that
  static types cannot express.
- Keep validation proportional. Validate external inputs and critical domain
  invariants, but do not add guards for internal states already guaranteed by
  types, constructors, or lifecycle rules.
- Keep policy separate from mechanism, especially in scheduling, cache
  allocation, execution, and backend selection.
- Treat execution plans as immutable contracts after scheduling.
- Prefer composition, dataclasses, and small focused objects. Add protocols or
  base classes only at real substitution boundaries.
- Do not create catch-all `utils.py`, `common.py`, or manager objects with
  unrelated responsibilities.
- Reject unsupported configurations with actionable errors. Never silently
  change semantics or hide an incorrect path behind a fallback.
- Keep accelerator-specific code behind backend boundaries.

## Correctness and Testing

- Run the test suite from the repository root with `python -m pytest`.
- Performance-sensitive algorithms should have a deterministic, readable
  reference implementation or trusted baseline for parity testing.
- Optimized paths need parity tests with documented numerical tolerances and
  should be independently disableable when practical.
- Prefer CPU-only unit tests for device-independent core logic.
- Test state transitions, ownership, and invariants, not only successful output.
- Update tests when behavior, state transitions, or public contracts change.
  Reproducible behavioral bug fixes should include regression tests when practical.
- Separate fast unit, GPU, distributed, and end-to-end tests so they can run
  independently.
- Do not treat plausible generated text as numerical or token-level validation.
- If validation cannot run locally, report exactly what was not verified and why.

## Performance and Dependencies

- Measure before optimizing and record workload, hardware, dtype, shapes, memory,
  and baseline; do not replace a correct, debuggable path with an unmeasured one.
- Keep required dependencies minimal and isolate optional accelerator or
  profiling dependencies from the base package.
- New dependencies require a clear rationale that considers correctness,
  maturity, maintenance cost, and implementation complexity.
- Do not introduce large distributed, service, or utility frameworks for narrow
  features without explicit agreement.

## Documentation and Changes

- Comments should explain invariants, ownership, reasoning, and tradeoffs rather
  than restating code.
- Keep detailed requirements and design explorations in local documentation; do
  not make tracked code depend on ignored documents being present.
- Make the smallest coherent change, preserve unrelated user work, and avoid
  expanding into adjacent or future subsystems without explicit scope.
- Call out changes to module ownership, dependency direction, or stable contracts
  before implementation.
