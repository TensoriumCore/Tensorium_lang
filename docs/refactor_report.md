# Tensorium Lang - Refactor Architecture Report

## 1) Executive Summary
- This audit was executed on branch `refactor/architecture-cleanup` with baseline validation (`cmake --build build -j` and `bash run_test.sh`) passing.
- Current architecture is functionally coherent end-to-end, but structurally concentrated in two large libraries: `tensorium` and `tensorium_mlir_backend`.
- The current layering leaks across boundaries:
  - frontend types leak into backend IR (`DomainIR` depends on `AST` types),
  - backend lowering depends directly on semantic analyzer concrete class,
  - validation over backend IR lives in `Sema` namespace/folder.
- A major risk for future maintainability is duplicated logic:
  - index-set rules (`i..n`) duplicated in several files,
  - MLIR index-attribute helper code duplicated across multiple passes,
  - repeated type-kind conversion/printing switches.
- Several files are currently placeholders/unwired (`TensoriumToLinalg`, `TsmOptimization`), increasing noise in ownership boundaries.
- There is at least one functional defect observed during audit: `ConTensor3` maps to `ConTensor4` in backend lowering (`lib/Backend/BackendBuilder.cpp:84`).
- Target architecture proposed in this report introduces explicit libraries per layer (`Core`, `AST`, `Lex`, `Parse`, `Sema`, `IR`, `Lowering`, `Runtime`, `MLIR IR`, `MLIR Semantic`, `MLIR Transforms`, `MLIR Codegen`, `Tools`).
- Refactor should be done incrementally with small compile-safe commits; no semantic behavior change except explicitly isolated bug fixes with tests.
- First phase (this branch) provides baseline artifacts and migration plan; next phases should split CMake targets and headers before moving files.

## 2) Etat Actuel (Architecture + Problemes)

### 2.1 Cibles CMake et modules reels
Current targets:
- Libraries: `tensorium`, `tensorium_mlir_backend`
- Executables: `Tensorium_cc`, `Tensorium_tester`
- TableGen: `TensoriumPassesIncGen`, `TensoriumOpsIncGen`

Top-level module map (logical):
- Front-end: `Lex`, `Parse`, `AST`, `Sema`
- Mid-end IR: `Backend/DomainIR` + `BackendBuilder`
- Runtime/eval: `Runtime`
- MLIR backend: `tensorium_mlir/*` (Dialect, Semantic, Transforms, Target/MLIRGen)
- Tooling: `tools/driver`, `tools/Tester`
- Tests: `tests/*.tn`, `tests/mlir/*.mlir`, `run_test.sh`

### 2.2 Points d'entree et flux d'execution principal
Primary CLI entrypoint:
- `tools/driver/main.cpp:100`

Main path from source text to outputs:
1. Read file -> `Lexer` (`tools/driver/main.cpp:193`)
2. Parse -> `Program AST` (`tools/driver/main.cpp:194`)
3. Semantic analysis -> indexed AST/type checks (`tools/driver/main.cpp:197`)
4. Lower to backend IR (`BackendBuilder::build`) (`tools/driver/main.cpp:240`)
5. Optional validation over backend IR (`tools/driver/main.cpp:242`)
6. Optional MLIR emission/pipeline (`tools/driver/main.cpp:297`)
7. Optional runtime execution (`tools/driver/main.cpp:305`)

### 2.3 Dependances observees (include + appel)
Observed dependency direction (module-level, simplified):
- `Lex -> Basic`
- `Parse -> Lex + AST`
- `Sema -> AST (+ Basic)`, and also `Sema/ProgramValidator -> Backend`
- `BackendBuilder -> AST + IndexedAST + Sema`
- `Runtime -> Backend`
- `MLIRGen -> Backend + MLIR Dialect/Passes`
- `Tools/driver -> AST + Lex + Parse + Sema + Backend + Runtime + MLIRGen`

Detected cycle at architecture level (hidden today by monolithic target):
- `Backend` depends on `Sema` (`include/tensorium/Backend/BackendBuilder.hpp:6`)
- `Sema` validation API depends on backend IR (`include/tensorium/Sema/ProgramValidator.hpp:2`)

### 2.4 Architectural smells
1. Monolithic target packing
- `tensorium` groups frontend, sema, backend IR builder, runtime.
- `tensorium_mlir_backend` groups dialect, semantic helpers, passes, codegen, init/pipeline.

2. Layering violations
- Backend IR leaks frontend type: `include/tensorium/Backend/DomainIR.hpp:3` + `TensorTypeDesc` fields (`:63`, `:97`).
- BackendBuilder API tied to concrete semantic analyzer (`include/tensorium/Backend/BackendBuilder.hpp:12`).
- Backend IR validator located under `Sema` namespace/folder (`include/tensorium/Sema/ProgramValidator.hpp:24`).

3. God files / mixed responsibilities
- `lib/Parse/Parser.cpp` (601 lines) parses expressions + declarations + simulation blocks.
- `lib/Sema/Sema.cpp` (441 lines) symbol setup + simulation validation + type checks + mode policy.
- `include/tensorium/Sema/tensor_type_checker.hpp` (560 lines, header-only heavy logic).
- `lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp` (328 lines) does IR emission + pass pipeline assembly.
- `tools/driver/main.cpp` (324 lines) includes argument parsing, orchestration, dump logic, runtime path.

4. Redundant or unused structure
- Empty placeholder files:
  - `include/tensorium_mlir/Conversion/TensoriumToLinalg/TensoriumToLinalg.h`
  - `lib/tensorium_mlir/Conversion/TensoriumToLinalg/TensoriumToLinalg.cpp`
  - `lib/tensorium_mlir/Dialect/Tensorium/Transforms/TsmOptimization.cpp`
- `registerAllPasses` currently empty (`lib/tensorium_mlir/Init/Passes.cpp:5`).

5. Defect found during audit
- `TensorKind::ConTensor3` lowered to `FieldKind::ConTensor4` (`lib/Backend/BackendBuilder.cpp:84`).

6. Local code-quality risk
- Uninitialized CLI flag: `bool dumpBackend, dumpBackendExpr = false;` (`tools/driver/main.cpp:103`).

### 2.5 Cartographie des responsabilites (niveau, invariants, dependances)

| Zone | Niveau | Responsabilite | Invariants essentiels | Dependances principales |
|---|---|---|---|---|
| `include/tensorium/Basic` | L0 | Tokens + signatures simples | `TokenType` stable, rank/variance coherents | STL only |
| `include/tensorium/AST`, `lib/AST` | L1 | AST syntaxique + printer | AST immuable par ownership `unique_ptr` | Basic/STL |
| `include/tensorium/Lex`, `lib/Lex` | L1 | Lexing DSL -> tokens | Position (line/col), keywords reconnus | Basic |
| `include/tensorium/Parse`, `lib/Parse` | L1 | Parsing tokens -> `Program` | Syntax validity, top-level ordering | Lex + AST |
| `include/tensorium/Sema`, `lib/Sema` | L1/L2 | Resolution symbolique + types + Einstein checks | Indices valides, variance coherente, mode exec/symbolic | AST (+ Backend for validator) |
| `include/tensorium/Backend/DomainIR.hpp` | L2 | IR interne cible runtime/MLIR | `FieldIR` up/down coherents, `ExprIR` typage | AST type leakage |
| `include/tensorium/Backend/BackendBuilder.hpp`, `lib/Backend` | L2 | Conversion AST+Sema -> DomainIR | Preserve semantics of equations/temps | AST + IndexedAST + Sema |
| `include/tensorium/Runtime`, `lib/Runtime` | L3 | Eval CPU scalaire 1D | Simulation required, Euler-only, scalar-only | Backend IR |
| `include/tensorium_mlir/Dialect/*`, `lib/.../IR` | L3 | Dialect/ops/types Tensorium MLIR | Op verifiers, rank/type contracts | MLIR core |
| `include/tensorium_mlir/Semantic`, `lib/.../Semantic` | L3 | Einstein index semantic helpers | role classification + validity | LLVM ADT |
| `lib/tensorium_mlir/.../Transforms` | L3 | Passes (analyze/canonicalize/lowering) | attrs `tin.idx.*` consistency | Dialect + Semantic |
| `include/tensorium_mlir/Target/MLIRGen.h`, `lib/.../MLIRGen.cpp` | L3 | DomainIR -> MLIR module + pipeline | conversion correctness + pass order | Backend IR + Dialect + Passes |
| `tools/driver` | L4 | CLI orchestration | deterministic option handling | all core libs |
| `tools/Tester` | L4 | Printer/test harness demo | parse/sema smoke checks | frontend libs |
| `tests/`, `run_test.sh` | L4 | Integration regression suite | expected pass/fail matrix | `Tensorium_cc` binary |

## 3) Architecture Cible (Modules + Regles)

### 3.1 Schema cible des modules et librairies CMake
Proposed CMake targets (incremental extraction):
- `tensoriumCore`
  - `include/tensorium/Core/*` (new): `SourceLocation`, diagnostics core, index utilities, string helpers/interner.
- `tensoriumAST`
  - `include/tensorium/AST/*`, `lib/AST/*` (AST nodes + visitors/printers).
- `tensoriumLex`
  - `include/tensorium/Lex/*`, `lib/Lex/*`.
- `tensoriumParse`
  - `include/tensorium/Parse/*`, `lib/Parse/*`.
- `tensoriumSema`
  - `include/tensorium/Sema/*` minus backend-IR validator.
- `tensoriumIR`
  - `include/tensorium/IR/*` (move from `Backend/DomainIR.hpp`).
- `tensoriumLowering`
  - `include/tensorium/Lowering/*` (move from `BackendBuilder`).
- `tensoriumValidation`
  - backend IR validation (move from `Sema/ProgramValidator`).
- `tensoriumRuntime`
  - `include/tensorium/Runtime/*`, `lib/Runtime/*`.
- `tensoriumMlirIR`
  - Tensorium dialect + ops + types.
- `tensoriumMlirSemantic`
  - Einstein semantic analysis helpers.
- `tensoriumMlirTransforms`
  - all MLIR transform passes.
- `tensoriumMlirCodegen`
  - MLIRGen + pipeline wiring.
- `tensoriumDriverLib` (optional but recommended)
  - CLI-independent orchestration service; `Tensorium_cc` becomes thin wrapper.

### 3.2 Regles de dependances (strict layering)
Mandatory direction:
- `Core` <- `AST` <- (`Lex`, `Parse`) <- `Sema` <- `IR` <- (`Lowering`, `Validation`) <- (`Runtime`, `MlirCodegen`) <- `Tools`

Constraints:
- `Lex/Parse` cannot include runtime/backend/mlir headers.
- `AST` must not include backend/runtime/mlir headers.
- `IR` must not include AST headers (replace leaked `TensorTypeDesc` with IR-native type metadata).
- `Sema` must not depend on `IR`.
- `MlirTransforms` depends on `MlirIR` + `MlirSemantic`, never on frontend parser/sema.
- `Tools` own orchestration; libraries own transformations.

### 3.3 Placement cible des composants demandes
- Fundamental types (`SourceLocation`, diagnostics, index alphabet, string interner): `tensoriumCore`.
- AST + visitors + printers: `tensoriumAST`.
- Sema (resolution, typing, symbol tables): `tensoriumSema`.
- Backend/lowering (DomainIR + AST->IR + MLIR pipeline): `tensoriumIR`, `tensoriumLowering`, `tensoriumMlir*`.
- Runtime/eval: `tensoriumRuntime`.
- Tools/CLI: `tools/driver`, `tools/Tester` linked against granular libs.
- Tests/fixtures:
  - keep `.tn` fixtures in `tests/`,
  - add `ctest` wrappers for parser/sema/runtime/mlir smoke bins.

## 4) Redondances Detectees + Actions

| Redondance (avant) | Apres propose | Risque | Test a ajouter |
|---|---|---|---|
| Index alphabet `{i..n}` duplicated in `Sema.hpp`, `CallSupport.cpp`, `ProgramValidator.cpp`, `tensor_type_checker.hpp` | Single utility in `tensoriumCore/IndexSet.h` (`isTensorIndex`, `isSpatialIndex`) | Low | Unit tests for accepted/rejected index names |
| MLIR attr helpers duplicated (`isAllStringAttrs`/`toRefs`/`fromRefs`) in 4 pass files | Shared helper in `tensorium_mlir/Dialect/Tensorium/Transform/AttrUtils.h/.cpp` | Low | Pass-level tests validating malformed attrs diagnostics |
| `parseSpecToIdx` duplicated in `EinsteinAnalyzeEinsumPass` and `EinsteinCanonicalizePass` | Single parser utility in `tensorium_mlir/Semantic/EinsteinSpec.{h,cpp}` | Medium | Canonicalization + analyze pass consistency test on same `spec` |
| `makeOffsets` and `getScalarFieldType` duplicated in stencil/dissipation passes | Shared stencil utility (`StencilUtils`) | Low | Stencil/dissipation regression tests with fixed expected offsets |
| Tensor kind mappings duplicated (`parseTensorTypeDesc`, `parseFieldDecl`, `deduceKind`, `lowerFieldKind`) | Central conversion table in `Core/TensorKinds` | Medium | Roundtrip tests kind<->(up,down)<->IR kind |
| Type print switches duplicated (`ASTPrinter.cpp`, `tools/Tester/Printer.cpp`) | Shared formatter helper `formatTensorKind()` | Low | Printer golden tests |
| Frontend/backend simulation enums duplicated and converted (`AST` vs `DomainIR`) | Promote shared enum types in `Core` or explicit conversion API in one place | Medium | Simulation config conversion tests |
| Validator placement mismatch (`Sema/ProgramValidator` validates backend IR) | Move to `Validation` module with IR-only API | Low | Existing `--validate` integration test + new unit test |

Specific defect to isolate in dedicated patch:
- `ConTensor3 -> ConTensor4` mapping in `lib/Backend/BackendBuilder.cpp:84`
  - Risk: High correctness risk for rank-3 contravariant tensors.
  - Add integration test: compile a `con_tensor3` evolution and check resulting IR rank/type.

## 5) Plan de Migration Incremente (Branch + Commits)

Planned commit sequence (one intention per commit):

1. `chore(refactor): add baseline snapshot notes and audit helper`
- But: freeze reproducible baseline before structural changes.
- Diff attendu: `docs/refactor_baseline.md`, `tools/dev/refactor_audit_snapshot.sh`.
- Risques: none.
- Validation: `cmake --build build -j`, `bash run_test.sh`.

2. `fix(build): align EinsteinLowering with current MLIR builder API` (already applied on this branch)
- But: keep branch compilable on LLVM/MLIR 20 and remove deprecated usage.
- Diff attendu: `lib/tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinLoweringPass.cpp`.
- Risques: low, no behavior change intended.
- Validation: build + full `run_test.sh`.

3. `refactor(cmake): introduce granular frontend targets`
- But: split monolith `tensorium` into `tensoriumCore`, `tensoriumAST`, `tensoriumLex`, `tensoriumParse`.
- Diff attendu: `lib/CMakeLists.txt`, new per-folder CMake files.
- Risques: link order/include visibility.
- Validation: build all + parser smoke.

4. `refactor(sema): extract tensor typing and index policy utilities`
- But: move index/type helpers out of header-heavy sema code.
- Diff attendu: new `include/tensorium/Core/*`, update `Sema`.
- Risques: subtle semantic regressions.
- Validation: all failing semantic tests remain failing for same reason.

5. `refactor(ir): create tensoriumIR and detach from AST types`
- But: remove `DomainIR -> AST` dependency.
- Diff attendu: move `DomainIR.hpp` to `include/tensorium/IR/`, replace `TensorTypeDesc` usage with IR-native types.
- Risques: conversion mismatches.
- Validation: backend dump comparison and runtime smoke.

6. `refactor(lowering): isolate AST+Sema -> IR in tensoriumLowering`
- But: remove direct `BackendBuilder` dependence on sema internals where possible.
- Diff attendu: `BackendBuilder` move/rename, API cleaned.
- Risques: orchestration call sites.
- Validation: CLI end-to-end compile.

7. `refactor(validation): move ProgramValidator to tensoriumValidation`
- But: remove Sema<->Backend layering cycle.
- Diff attendu: move `ProgramValidator.*`, update includes/namespaces.
- Risques: namespace/API churn.
- Validation: `--validate` behavior unchanged.

8. `refactor(runtime): link against tensoriumIR only`
- But: enforce runtime boundary.
- Diff attendu: runtime includes and target links.
- Risques: hidden dependencies.
- Validation: `--run-cpu` smoke test.

9. `refactor(mlir): split mlir backend into IR/Semantic/Transforms/Codegen targets`
- But: shrink API surface and ownership by concern.
- Diff attendu:
  - `tensoriumMlirIR`
  - `tensoriumMlirSemantic`
  - `tensoriumMlirTransforms`
  - `tensoriumMlirCodegen`
- Risques: registration/link errors.
- Validation: `--dump-mlir` suite from `run_test.sh`.

10. `refactor(passes): deduplicate Einstein/stencil helper utilities`
- But: remove duplicated pass logic and stabilize attrs contract.
- Diff attendu: new shared helper files + pass simplification.
- Risques: pass behavior drift.
- Validation: add pass-focused regression fixtures.

11. `refactor(driver): extract orchestration library and thin CLI`
- But: reduce `main.cpp` god object and centralize pipeline flow.
- Diff attendu: new `tensoriumDriverLib`, smaller `tools/driver/main.cpp`.
- Risques: CLI option behavior drift.
- Validation: CLI snapshot tests.

12. `chore(test): add CTest wiring and non-regression matrix`
- But: CI-friendly granular validation.
- Diff attendu: `enable_testing()`, `add_test(...)`, wrapper scripts.
- Risques: flaky env assumptions.
- Validation: `ctest --output-on-failure`.

### Cibles CMake a creer/adapter explicitement
- New: `tensoriumCore`, `tensoriumAST`, `tensoriumLex`, `tensoriumParse`, `tensoriumSema`, `tensoriumIR`, `tensoriumLowering`, `tensoriumValidation`, `tensoriumRuntime`, `tensoriumMlirIR`, `tensoriumMlirSemantic`, `tensoriumMlirTransforms`, `tensoriumMlirCodegen`, `tensoriumDriverLib`.
- Adapt:
  - `Tensorium_cc` links `tensoriumDriverLib` (+ runtime/mlir as needed).
  - `Tensorium_tester` links `tensoriumAST + tensoriumLex + tensoriumParse + tensoriumSema`.
  - Keep legacy aggregate aliases temporarily (`tensorium`, `tensorium_mlir_backend`) during transition if needed.

## 6) Risques & Mitigations
- Risk: accidental behavior changes during module split.
  - Mitigation: strict compile/test gate per commit; no mixed-intent commits.
- Risk: include visibility/link regressions after target split.
  - Mitigation: define `PUBLIC/PRIVATE` include policy per target and add linker smoke tests.
- Risk: semantic drift in Einstein passes after utility extraction.
  - Mitigation: add pass golden tests for `tin.idx.*` attrs.
- Risk: performance impact from abstraction.
  - Mitigation: keep hot paths in `.cpp`, avoid virtual indirection where unnecessary.
- Risk: migration stall due broad scope.
  - Mitigation: keep compatibility aliases temporarily and track TODO removals with explicit tickets.

## 7) Validation Checklist
- Build:
  - `cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/llvm-20 -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-O0 -g"`
  - `cmake --build build -j`
- Tests:
  - `bash run_test.sh`
  - Add `ctest --output-on-failure` once CTest integration is introduced.
- API surface:
  - no direct high-level include from lower layers (`IR`/`Runtime` must not include parser/sema).
- Style/quality:
  - run formatter/lint used by project (if added later).
  - add include hygiene check (IWYU or equivalent) once target split is stable.
- Regression focus:
  - semantic error fixtures (`tests/*error*.tn`) must keep expected failures.
  - MLIR dump path (`--dump-mlir`) must preserve pass validity checks.

## Assumptions Explicites
- LLVM/MLIR 20 is the active toolchain.
- Current official regression command is `bash run_test.sh` (no CTest yet).
- Refactor target is behavior-preserving, except isolated bug fixes covered by tests.

## 8) Phase Execution Update

### Phase 1 done
- Fixed backend lowering defect:
  - `TensorKind::ConTensor3` now maps to `FieldKind::ConTensor3` in `lib/Backend/BackendBuilder.cpp`.
- Added non-regression test:
  - `tools/Tester/UnitTests.cpp` validates backend kind for a `con_tensor3` field.
  - Behavior demonstrated:
    - before fix: test failed (`expected backend kind ConTensor3`),
    - after fix: test passes.
- Fixed driver hygiene:
  - initialized `dumpBackend` explicitly in `tools/driver/main.cpp`.
- Regression suite now executes internal unit tests from `run_test.sh`.

### Layering check added
- Added mechanical boundary checker: `tools/dev/check_layering.sh`.
- Rule coverage:
  - forbid `DomainIR` -> `AST` includes,
  - forbid `lib/Runtime` -> `Parse/Sema`,
  - forbid `lib/Parse` -> `Runtime/Backend/MLIR`,
  - forbid `include/tensorium/Validation` and `lib/Validation` -> `AST/Parse/Sema`.
- Status:
  - script exits `0` on current `HEAD`,
  - usage documented in `docs/refactor_baseline.md`.

### IndexSet extracted
- Added shared index policy utility: `include/tensorium/Core/IndexSet.h`.
- Replaced duplicated index helpers in:
  - `include/tensorium/Sema/Sema.hpp`,
  - `lib/Sema/Sema.cpp`,
  - `lib/Sema/CallSupport.cpp`,
  - `lib/Validation/ProgramValidator.cpp`,
  - `include/tensorium/Sema/tensor_type_checker.hpp`.
- Extended unit tests with accepted/rejected index checks in
  - `tools/Tester/UnitTests.cpp`.
- Validation status:
  - `cmake --build build -j` passes,
  - `bash run_test.sh` passes.

### Phase 2 done
- IR detached from frontend AST:
  - introduced `include/tensorium/IR/TensorType.hpp` (`tensorium::ir::TensorType`),
  - `include/tensorium/Backend/DomainIR.hpp` now consumes IR-native tensor type and no longer includes AST headers.
- Boundary conversions implemented at lowering edge:
  - `lib/Backend/BackendBuilder.cpp` converts frontend `TensorTypeDesc` into IR tensor type,
  - MLIR and runtime consumers now read IR-native tensor metadata only.
- Validation module moved out of Sema:
  - `ProgramValidator` moved to `include/tensorium/Validation/ProgramValidator.hpp` and `lib/Validation/ProgramValidator.cpp`,
  - namespace/API moved from `tensorium::sema` to `tensorium::validation`,
  - `tools/driver/main.cpp` updated to call validation in the same pipeline position.
- Verification:
  - `cmake --build build -j` passes after each commit,
  - `bash run_test.sh` passes after the Phase 2 commit blocks,
  - `bash tools/dev/check_layering.sh` is green on `HEAD`.

### Phase 3 started (CMake split scaffolding)
- Introduced explicit layered targets in `lib/CMakeLists.txt`:
  - `tensoriumCore` (frontend + sema + runtime sources),
  - `tensoriumIR` (IR headers as dedicated interface target),
  - `tensoriumLowering` (AST/Sema -> IR lowering),
  - `tensoriumValidation` (IR validation module).
- Kept compatibility target:
  - `tensorium` is now an aggregate interface linking the four targets above,
  - compatibility is preserved for downstream consumers that still link `tensorium`.
- Tool linkage advanced to explicit layers:
  - `tools/driver/CMakeLists.txt` links `tensoriumLowering` + `tensoriumValidation` (+ MLIR backend),
  - `tools/Tester/CMakeLists.txt` links `tensoriumLowering` + `tensoriumValidation`.
- Validation:
  - `cmake --build build -j` passes,
  - `bash run_test.sh` passes,
  - `bash tools/dev/check_layering.sh` passes.

### Phase 3 continued (MLIR layered targets)
- Split monolithic MLIR backend target into explicit layers in `lib/CMakeLists.txt`:
  - `tensoriumMlirIR` (dialect/types/ops),
  - `tensoriumMlirSemantic` (Einstein semantic helpers),
  - `tensoriumMlirTransforms` (all Tensorium transform passes),
  - `tensoriumMlirCodegen` (MLIRGen + registry + pipeline).
- Kept compatibility facade:
  - `tensorium_mlir_backend` remains available as an interface target for compatibility.
- Driver linkage now uses explicit codegen target:
  - `tools/driver/CMakeLists.txt` links `tensoriumMlirCodegen` directly.
- Validation:
  - `cmake --build build -j` passes after each commit,
  - `bash run_test.sh` passes after the commit block,
  - `bash tools/dev/check_layering.sh` passes.

### Phase 3 continued (MLIR warning cleanup)
- Removed deprecated MLIR builder usage in active code paths:
  - migrated `OpBuilder::create<...>` calls to `OpTy::create(...)` in:
    - `lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp`,
    - `lib/tensorium_mlir/Dialect/Tensorium/Transforms/StencilLoweringPass.cpp`,
    - `lib/tensorium_mlir/Dialect/Tensorium/Transforms/DissipationPass.cpp`,
    - `lib/tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinCanonicalizePass.cpp`.
- Result:
  - no remaining `.create<...>` calls in `lib/tensorium_mlir`,
  - build remains green with unchanged behavior.

## 9) Semantic correctness audit

### Findings by severity
- S0 (incorrect math risk): contraction and derivative semantics were partially implicit.
  - Before this phase, Einstein contraction information was not represented explicitly in backend IR and differential calls were mostly generic function calls.
  - Evidence points:
    - prior contraction/type checks were spread in `include/tensorium/Sema/tensor_type_checker.hpp`; now centralized with `IndexAnalysis` (`include/tensorium/Sema/tensor_type_checker.hpp:32`, `include/tensorium/Sema/tensor_type_checker.hpp:140`, `include/tensorium/Sema/tensor_type_checker.hpp:231`).
    - backend IR now carries explicit tensor ops (`include/tensorium/Backend/DomainIR.hpp:60`, `include/tensorium/Backend/DomainIR.hpp:115`, `include/tensorium/Backend/DomainIR.hpp:154`, `include/tensorium/Backend/DomainIR.hpp:168`).
- S1 (ambiguity): collisions between free and bound indices could degrade into less actionable assignment mismatch diagnostics.
  - Now normalized to explicit collision diagnostics in assignment checking (`include/tensorium/Sema/tensor_type_checker.hpp:553` onward).
  - `d_i`, `nabla_i`, `covariant_derivative`, `grad`, `div` builtins are explicitly recognized in executable mode (`lib/Sema/CallSupport.cpp:6`).
- S2 (technical debt): analysis/transformation boundaries were blurry across Sema/lowering.
  - This phase enforces a clearer split:
    - Sema: IndexAnalysis + validation (`include/tensorium/Sema/tensor_type_checker.hpp:140`).
    - IR: explicit op forms (`include/tensorium/Backend/DomainIR.hpp:60`).
    - Lowering: explicit op materialization and conversion (`lib/Backend/BackendBuilder.cpp:118`, `lib/Backend/BackendBuilder.cpp:159`, `lib/Backend/BackendBuilder.cpp:189`).

### Design decisions (explicit vs implicit)
- Einstein notation remains source-level friendly, but contractions are now materialized in backend IR as explicit `ContractionIR` with `summedIndices`.
- Tensor multiplication for tensor operands is represented as `TensorProductIR`; contraction is a distinct step.
- Differential operations are explicit in IR:
  - `PartialDerivativeIR`, `GradientIR`, `CovariantDerivativeIR`, `DivergenceIR`.
- Covariant derivative policy:
  - accepted only when a connection tensor (`Gamma`/`GammaU`/`Christoffel`, rank-3) is present (`lib/Sema/Sema.cpp:250`, `lib/Sema/Sema.cpp:354`),
  - otherwise rejected with stable diagnostic.

### Diagnostics examples
- Invalid contraction variance:
  - `Tensorium error: Implicit contraction of index 'i' requires explicit metric or inverse metric`
  - fixture: `tests/semantic/einstein/02_invalid_variance_contraction.tn`
- Index collision (free vs bound):
  - `Tensorium error: Index collision: symbol 'i' is both free and bound; rename one index in RHS.`
  - fixture: `tests/semantic/einstein/05_capture_requires_rename.tn`
- Missing connection for covariant derivative:
  - `Tensorium error: Covariant derivative requires connection tensor Gamma (rank-3 field)`
  - fixture: `tests/semantic/diff/04_covariant_without_gamma_error.tn`

### Schwarzschild 2D/3D status
- Added regression fixtures:
  - `tests/fixtures/gr/schwarzschild_2d.tn`
  - `tests/fixtures/gr/schwarzschild_3d.tn`
- Added structural checks in regression runner (`run_test.sh`):
  - fixture validates successfully,
  - backend IR dump must contain explicit `contraction(...)` and `partial_...(...)`.
- Optional benchmark scaffold added:
  - `tools/Bench/bench_schwarzschild.sh` logs validate/codegen timing in `/tmp/tensorium_bench`.

## 10) Semantic phase update
- Added dedicated semantic test packs:
  - Einstein: `tests/semantic/einstein/*`
  - Differential ops: `tests/semantic/diff/*`
- `run_test.sh` now includes:
  - acceptance/rejection checks for semantic Einstein and differential cases,
  - stable diagnostic substring checks for key semantic failures,
  - Schwarzschild fixture structural non-regression checks.
- MLIR lowering now consumes explicit backend ops for:
  - contraction/tensor-product/index operations (`lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp:184`),
  - differential operations (`lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp:216`).

## 11) Phase 3 Update (Ops Split + Canonical + Schwarzschild Baseline)

### 3.A Ops split completed
- `include/tensorium/Backend/DomainIR.hpp` is now a compatibility umbrella only.
- IR definitions were split into dedicated headers:
  - `include/tensorium/IR/IRBase.hpp`
  - `include/tensorium/IR/EinsteinOps.hpp`
  - `include/tensorium/IR/DifferentialOps.hpp`
  - `include/tensorium/IR/IRPrinter.hpp`
- Legacy include path kept stable:
  - `include/tensorium/Backend/IRPrinter.hpp` now forwards to `include/tensorium/IR/IRPrinter.hpp`.

### 3.B Canonical passes + verifier
- Added dedicated IR canonicalization passes (outside Sema):
  - `validation::canonicalizeDifferentialIR` in `lib/IR/CanonicalizeDiff.cpp`
  - `validation::canonicalizeEinsteinIR` in `lib/IR/CanonicalizeEinstein.cpp`
- Added IR verifier module:
  - `validation::verifyIR` in `lib/IR/IRVerifier.cpp`
  - API exposed by `include/tensorium/Validation/IRVerifier.hpp`
- Driver pipeline now runs:
  1. AST/Sema -> backend IR lowering
  2. Differential canonicalization
  3. Einstein canonicalization
  4. IR verifier
  5. Existing semantic/IR validation and optional MLIR/runtime paths
- Canonical invariants enforced:
  - `GradientIR` and `DivergenceIR` are sugar and must not survive verifier.
  - Contraction summed indices must be canonicalized (sorted + unique + valid).
  - Covariant derivative must carry a valid derivative index and connection availability.

### 3.B tests added (structured IR checks)
- Added canonical fixtures:
  - `tests/ir/canonical/01_gradient_sugar.tn`
  - `tests/ir/canonical/02_divergence_sugar.tn`
  - `tests/ir/canonical/03_trace_from_contract.tn`
- Extended `tools/Tester/UnitTests.cpp` with structured IR assertions (not fragile text matching):
  - gradient sugar -> `PartialDerivativeIR`
  - divergence sugar -> `ContractionIR(CovariantDerivativeIR(...))`
  - contract-trace canonicalization
  - alpha-renaming insertion on risky index capture
  - verifier rejection of uncanonicalized differential sugar

### 3.C Schwarzschild canonical + perf baseline
- Schwarzschild fixtures are now covered by canonical IR structural checks in unit tests:
  - `tests/fixtures/gr/schwarzschild_2d.tn`
  - `tests/fixtures/gr/schwarzschild_3d.tn`
  - assertions: canonical IR contains contraction + partial derivative nodes, and no residual gradient/divergence sugar.
- Bench script updated:
  - `tools/Bench/bench_schwarzschild.sh` now writes timestamped logs into `tools/Bench/out/<timestamp>/`.
  - `.gitignore` updated to ignore bench artifacts under `tools/Bench/out/`.
- Current baseline sample (`20260211_145648`):
  - Schwarzschild 2D: validate ~0.02s, backend dump ~0.01s, MLIR codegen ~0.02s
  - Schwarzschild 3D: validate ~0.01s, backend dump ~0.01s, MLIR codegen ~0.02s

### Hotspots and next micro-optimizations
- Hotspot candidates observed in the new IR stage:
  - repeated recursive scans over expression trees in canonicalization and verifier,
  - repeated index usage collection/allocation in Einstein canonicalization,
  - repeated string allocations for index names.
- Next measured optimizations to prioritize:
  1. cache per-expression index-use summaries during one canonical pass traversal,
  2. switch index representation from `std::string` to compact interned/char IDs in IR nodes,
  3. reduce temporary allocations in canonical passes (scratch buffers reused per evolution).

## 12) Phase 4/5 Update (Decompose + Init 3+1)

### 12.A Operations and semantics
- Added dedicated MLIR op:
  - `tensorium.decompose3p1_from_metric(%g) -> (%alpha,%beta,%gamma,%gammaU)`.
- Kept `tensorium.init3p1(...)` as explicit binding/normalization op for downstream use.
- Removed semantic ambiguity:
  - no legacy `tensorium.split3p1` op in emitted MLIR.

### 12.B Lowering behavior
- Metric path (`metric4` present):
  1. emit `tensorium.metric4` with 16 SSA operands,
  2. emit `tensorium.decompose3p1_from_metric(metric)`,
  3. emit `tensorium.init3p1(decompose results)` and bind mapped fields.
- Decomposed path (`alpha/beta/gamma/gammaU` present):
  - emit `tensorium.init3p1` directly from decomposed expressions.
- `split_3p1` mapping validation now accepts:
  - metric-based initial data,
  - or decomposed initial data.

### 12.C Supported decomposition scope
- `decompose3p1_from_metric` accepts symmetric metrics with optional
  spatial cross terms (`g_ij`) and time-space cross terms (`g_ti`).
- Current explicit guardrails:
  - non-symmetric metric components are rejected with
    `"decompose3p1_from_metric requires symmetric metric components"`.
- Current numeric semantics in front init evaluator:
  - `gamma_ij = g_ij`,
  - `beta_i = g_{0i}`,
  - `gammaU = inverse(gamma)` (diag fast path + 3x3 inverse fallback),
  - `alpha = sqrt(beta_i beta^i - g_tt)`.

## 13) Schwarzschild MLIR verification

### 13.A Structural proof (robust, SSA-level)
- Structural validation is now implemented in C++ unit tests (MLIR IR walk),
  not via fragile shell text-grep.
- Invariants locked by `tools/Tester/UnitTests.cpp`:
  - `@tensorium_init` contains `metric4 + decompose3p1_from_metric + init3p1`
    and uses `tensorium.assign` (no `tensorium.dt_assign`).
  - `@tensorium_rhs` contains `tensorium.dt_assign` and excludes
    `metric4/decompose3p1_from_metric/assign`.
  - `@tensorium_entry` contains exactly 2 calls in order:
    `@tensorium_init` then `@tensorium_rhs` (plus return).
  - use-def bridge from init to rhs is checked through entry call operands:
    fields assigned in init are required to be read in rhs.
  - RHS checks remain structural:
    `gammaU` must feed a `mul` used by `contract`,
    and `alpha*gamma` must be used in the `dt K` path.
- Added a negative invariant test:
  - injects a `metric4` op into `@tensorium_rhs` and asserts invariant rejection.

### 13.B Optimization baseline (minimal CSE/const-fold)
- Enabled CSE/canonicalization effectiveness by marking pure producers:
  - `tensorium.const`, `tensorium.param`, `tensorium.coord` now `Pure`.
- Pipeline continues to run `canonicalizer` + `CSE`.
- Added structural assertion for Schwarzschild:
  - `2*M/r` subexpression appears once after pipeline normalization.

### 13.C New regression/negative tests
- Added initial-data fixtures:
  - `tests/semantic/initial_data/offdiag_metric.tn` is expected to pass
    (symmetric spatial cross term `g_ij`).
  - `tests/semantic/initial_data/04_nonsymmetric_metric_not_supported.tn`
    is expected to fail with the non-symmetric metric diagnostic.
  - `tests/semantic/initial_data/05_shift_metric_not_supported.tn` is
    expected to fail when `g_ti` is non-zero (`beta` unsupported in
    `decompose3p1_from_metric`).
- `run_test.sh` no longer performs fragile MLIR-grep for init/rhs architecture;
  this is enforced by structural unit tests.

## 14) Init/RHS Split Hardening

### 14.A Signature minimization
- Function signatures are now data-driven from actual field usage:
  - `@tensorium_init` receives only fields needed for init writes/reads.
  - `@tensorium_rhs` receives only fields referenced by RHS equations.
  - `@tensorium_entry` keeps the full program-field signature and forwards the
    minimal subsets to each callee.
- Forwarding order is deterministic and stable:
  - argument order follows `module.fields` declaration order,
  - each callee receives an ordered subsequence of that list (no padding).
- Schwarzschild 3D concrete result:
  - before: init/rhs both took all 8 field arguments.
  - after: `@tensorium_init` takes 3 args (`alpha`, `gamma`, `gammaU`);
    `@tensorium_rhs` takes 6 args (`alpha`, `phi`, `H`, `gamma`, `gammaU`, `K`).

### 14.B Rationale
- Keeps init-time and rhs-time concerns separated in both op placement and API.
- Reduces accidental coupling (unused args cannot be read/written by construction).
- Preserves semantics: `init3p1` is retained (still explicit in init path),
  while stores remain split between `assign` (init) and `dt_assign` (rhs).

### 14.C IR invariants (init/rhs)
- RHS write path guard:
  - `@tensorium_rhs` must not contain init-only ops
    (`metric4`, `decompose3p1_from_metric`, `init3p1`, `assign`),
    and may write only via `dt_assign`.
- GammaU provenance guard:
  - In Schwarzschild structural verification, the `gammaU` value used by
    `contract(...)` must come from a `tensorium.ref` sourcing a field that was
    assigned by `@tensorium_init` through entry-call forwarding.
  - `@tensorium_rhs` is rejected by test if it constructs local `gammaU`
    tensors (e.g. `build_con_tensor2`) for the contraction path.
- Front-end guard:
  - Non-`dt` writes to declared fields inside `evolution` are rejected
    semantically (`Cannot redeclare field ... as local`).

## 15) MLIR normalization passes

- Normalization is applied **after MLIRGen module construction** in
  `tensorium_mlir::buildMLIRModule(...)` (`lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp`).
- Post-MLIRGen normalization pipeline is now explicit and configurable through
  `MLIRGenOptions`:
  - `enableMLIRCanonicalizePass` (default: `true`)
  - `enableMLIRCSEPass` (default: `true`)
  - `enableMLIRInlinePass` (default: `false`, optional)
- Test policy:
  - Unit/integration tests run with canonicalize + CSE enabled by default
    (stable compact MLIR dumps),
  - inline remains optional and off unless explicitly requested.
- Structural regression test coverage:
  - a dedicated UnitTests case compares Schwarzschild MLIR with/without
    normalization and asserts compaction (`2*M/r` and `sin(theta)` duplicates
    are reduced to single producers),
  - init/rhs invariants remain validated on normalized MLIR.

## 16) Einstein canonical normal form

- Canonical Einstein form is now enforced at backend IR level before MLIR
  emission (`validation::canonicalizeEinsteinIR`):
  - sorted/unique contraction summed indices,
  - deterministic alpha-renaming of dummy indices to canonical names
    (excluding currently free indices),
  - redundant `index_rename` elimination by applying renaming directly into
    the expression tree,
  - redundant `index_permute` elimination (`index_permute(index_permute(x))`
    with identical order, and empty order),
  - `trace(...)` canonicalized into contraction form.
- Canonicalization is idempotent by test:
  - applying Einstein canonicalization twice preserves the same canonical
    expression key/signature.
- Equivalent Einstein DSL fixtures now checked for MLIR canonical equivalence:
  - `tests/semantic/einstein/canon/01_contract_ij.tn`
    vs `tests/semantic/einstein/canon/02_contract_mn.tn`.
- Validation strategy is structural (UnitTests IR/MLIR walk), not textual grep.

## 17) Christoffel Front Contract

- Added executable builtin `christoffel(gamma, gammaU)` in front-end typing/lowering.
- Typing contract:
  - arg0 must be covariant rank-2 (`0,2`),
  - arg1 must be contravariant rank-2 (`2,0`),
  - result is mixed rank-3 (`1,2`), represented as `!tensorium.field<f64,1,2>`.
- DSL field declarations now accept `mixed_tensor(up=...,down=...)` for mixed variance fields.
- Lowering strategy (no magic MLIR op):
  - builtin is expanded in backend IR into explicit Einstein-form expression
    using only `PartialDerivative`, `Binary(+/-/*)`, `TensorProduct`, and
    `Contraction`,
  - MLIR emitted form uses only `tensorium.deriv`, `tensorium.add/sub/mul`,
    and `tensorium.contract` (then existing canonicalize/CSE pipeline).
- Numeric anti-false-green coverage:
  - `tests/fixtures/gr/schwarzschild_christoffel_3d.tn` + UnitTests verify
    Schwarzschild reference values at `M=1, r=10, theta=pi/2`:
    `Gamma^r_rr`, `Gamma^r_thetatheta`, `Gamma^r_phiphi`,
    `Gamma^theta_rtheta`, `Gamma^phi_rphi`, `Gamma^phi_thetaphi`,
  - structural MLIR test asserts Christoffel path contains deriv/add/sub/mul/contract
    and that `gammaU` feeding contraction comes from init-assigned field provenance.

## 18) RHS MLIR Anti-Bias Evaluation

- Added a dedicated front-only RHS evaluator:
  - API: `include/tensorium_mlir/Target/MLIRGen/RhsEvaluator.h`
  - Impl: `lib/tensorium_mlir/Target/MLIRGen/RhsEvaluator.cpp`
- Scope (generic op subset, no Schwarzschild hardcode):
  - `tensorium.ref` (indices + offsets),
  - `tensorium.deriv` (central FD, interior-only),
  - `tensorium.add/sub/mul/div`,
  - `tensorium.contract` (sum over `sum_indices`),
  - `tensorium.promote`,
  - `tensorium.dt_assign`.
- Test upgrade:
  - `testSchwarzschildChristoffelNumericPoint` now executes `@tensorium_rhs`
    emitted from `tests/fixtures/gr/schwarzschild_christoffel_3d.tn` on a
    small 3D grid, then checks six analytical Christoffel components at the
    center point (`M=1, r=10, theta=1`).
  - This removes the prior bias where Christoffel numeric checks were evaluated
    through a dedicated backend-expression helper instead of RHS MLIR execution.
- Driver instrumentation (without breaking existing suite defaults):
  - New MLIR flags:
    - `--mlir-disable-threading`
    - `--mlir-print-op-on-diagnostic`
    - `--mlir-print-ir-after-failure`
    - `--mlir-strict-pipeline`
  - `emitMLIR(...)` now returns pipeline success.
  - Driver no longer prints `[Tensorium] OK` for files where MLIR pipeline fails;
    it prints `[Tensorium] FAILED` (and returns non-zero if `--mlir-strict-pipeline` is set).

## 19) Lowered-Only Module Cleanup

- Added a dedicated transform pass:
  - `createTensoriumStripSourceFuncsPass()`
  - implementation: `lib/tensorium_mlir/Dialect/Tensorium/Transforms/StripSourceFuncsPass.cpp`
- Purpose:
  - after grid lowering passes have produced executable kernels, remove source-level
    `tensorium_init`, `tensorium_rhs`, and `tensorium_entry` so the module is closer
    to LLVM-convertible form.
- Safety contract:
  - `tensorium_init` is removed only when an init replacement exists
    (`tensorium_init_point` or `tensorium_init_grid_*`),
  - `tensorium_rhs` is removed only when an RHS replacement exists
    (`tensorium_rhs_grid_*`),
  - `tensorium_entry` is removed only when both init and RHS replacements exist.
- Driver wiring:
  - new flag: `--tensorium-strip-source-funcs`
  - this flag is explicit and opt-in (default behavior unchanged).
- Structural test coverage:
  - `testStripSourceFuncsAfterGridLowering` asserts:
    - source functions are removed,
    - lowered affine grid kernels are present,
    - no `tensorium.*` operations remain in the resulting module.
