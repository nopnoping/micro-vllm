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
- `config` owns shared immutable configuration contracts.
- `core` owns device-independent scheduling and logical cache state.
- `executor` owns execution interfaces and worker lifecycle coordination.
- `worker` owns device resources and turns execution plans into model runs.
- `attention` owns attention semantics, runtime metadata, tensor operators, and
  operator selection.
- `models` owns the architecture registry, model assembly, and loading tensor
  streams into model parameters.
- `model_loader` owns user model resolution, config file parsing, remote
  download, checkpoint discovery, tensor iteration, and load orchestration.
- `layers` owns reusable, architecture-independent neural-network layers.
- `sampling` owns sampling configuration and logits-to-token behavior.

Enforce these boundaries:

- Core code must not invoke model kernels or own device tensors.
- The scheduler decides which requests run or are preempted; the KV cache manager
  owns logical block allocation, sharing, and release mechanisms.
- Executors and model runners must not mutate scheduler-owned request state.
- Models and layers must not allocate, free, or preempt logical cache blocks.
- Models must not access external config files, model paths, networks, Hub APIs,
  or checkpoint files.
- Model loaders must not import concrete models or interpret model-specific
  config fields.
- Model loaders must not interpret architecture-specific parameter layouts;
  they pass tensor streams to the selected model.
- Model runners own model construction/loading invocation, placement, warmup,
  memory profiling, physical KV tensor allocation, execution, and those device
  resources' lifecycles.
- Workers only orchestrate device processes and model-runner lifecycle; they do
  not construct, load, place, warm up, profile, retain models, or own model KV
  tensors directly.
- Model runners expose the profiled physical block capacity to the control
  plane; the control plane owns logical block IDs and lifecycle state.
- Shared schemas must not import concrete implementations or create dependency
  cycles.

## Design Rules

- Pass configuration and runtime context explicitly. Avoid mutable global state,
  hidden singletons, duplicate counters, and derived phase flags that can diverge.
- Rely on type annotations and static checking for parameter types. Reserve
  runtime validation for domain values, state transitions, and invariants that
  static types cannot express.
- Assume typed internal values and structures satisfy their contracts. Add
  runtime validation only for user-provided inputs and externally sourced domain
  values; do not repeat defensive checks while forwarding internal data between
  components.
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
