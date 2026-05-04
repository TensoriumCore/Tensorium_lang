# Refactor Baseline Snapshot

Date: 2026-02-11  
Branch: `refactor/architecture-cleanup`

## Scope
- Capture an objective baseline before architecture refactor commits.
- Record build/test commands used for non-regression checks.

## Commands Executed

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure
bash run_test.sh
```

## Result Summary
- Build: `OK`
- Main binary target built: `Tensorium_cc`
- CTest integration: `tensorium_unittests` and `tensorium_full_suite`
- Full test script: `ALL TESTS PASSED`
- Test harness output path: `/tmp/tensorium_tests`

## Notes
- Several test runs intentionally print `Pipeline failed` for cases expected by the current suite logic; `run_test.sh` still exits successfully.
- `ctest` delegates the complete integration suite to `run_test.sh`; keep `run_test.sh` as the source of truth for full regression coverage.

## Snapshot Helper
- Helper script added for future snapshots: `tools/dev/refactor_audit_snapshot.sh`
- Script extracts:
  - CMake targets
  - Declared libraries/executables
  - `lib/*.cpp` files not wired in `lib/CMakeLists.txt`

## Layering Check Helper
- Layering guard script: `tools/dev/check_layering.sh`
- Usage:

```bash
bash tools/dev/check_layering.sh
```

- Current expected status (pre-architecture split):
  - script passes on HEAD.
  - no whitelist / expected-fail behavior is used.
