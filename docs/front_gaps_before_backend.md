# Front Gaps Before Backend (Schwarzschild Init-Only Milestone)

## Scope
- Branche auditée: `refactor/architecture-cleanup`.
- Objectif: verrouiller ce qui manque côté front (`Tensorium_lang` + IR + MLIRGen) avant backend/JIT.
- Périmètre: **init Schwarzschild uniquement** (`@tensorium_init`), sans lowering backend complet.
- Source de vérité technique utilisée: code actuel + dump MLIR de `tests/fixtures/gr/schwarzschild_3d.tn` (`/tmp/schw3d_initrhs.mlir`).

## Definition Of Done (front-view) for JIT Milestone
Le jalon "Schwarzschild numeric init-only" est considéré prêt côté front quand:
- la fonction exécutable cible est `@tensorium_init` (pas `@tensorium_entry`),
- les entrées runtime minimales sont explicites et stables:
  - paramètres: `M`,
  - coordonnées par point: `r`, `theta`, `phi` (phi optionnel selon dimension/config),
  - description de grille: `N` points + mapping index->coordonnées,
- les sorties sont explicites:
  - minimum recommandé: `alpha`, `gamma_ij`, `gammaU^ij`,
  - optionnel: `g_{mu,nu}` si on garde une sortie métrique brute,
- un contrat mémoire est défini (format SoA recommandé, cf. section D),
- les valeurs numériques attendues Schwarzschild sont vérifiées au moins sur un point de référence,
- la sémantique MLIR des ops init est définie (pas seulement "op présente").

## Minimal Schwarzschild Numeric Example
Point de référence demandé: `M=1`, `r=10`, `theta=pi/2`.

Formules:
- `f = 1 - 2*M/r = 0.8`
- `g_tt = -f = -0.8`
- `g_rr = 1/f = 1.25`
- `g_thetatheta = r^2 = 100`
- `g_phiphi = r^2 * sin^2(theta) = 100`
- `alpha = sqrt(f) = 0.8944271909999159`
- `gamma_ij = diag(1.25, 100, 100)`
- `gammaU^ij = diag(0.8, 0.01, 0.01)`

Ces valeurs doivent devenir des assertions de non-régression au jalon backend/JIT.

## A) Gap audit: Semantics / AST / DSL
- `initial_data`/`metric4`/`split_3p1` est bien supporté côté parse/AST:
  - `lib/Parse/Parser.cpp:400`, `lib/Parse/Parser.cpp:511`, `include/tensorium/AST/AST.hpp:178`.
- La validation Sema de structure est en place:
  - dimensions/symétrie/coordonnées: `lib/Sema/Sema.cpp:378`, `lib/Sema/Sema.cpp:404`, `lib/Sema/Sema.cpp:333`.
- Le paramètre `M` est implicite (pas de déclaration DSL dédiée):
  - un `VarExpr` inconnu est transformé en `IndexedVarKind::Parameter` (`lib/Sema/Sema.cpp:225`).
  - Conséquence: pas de contrat explicite "params requis", pas de typage nominal de paramètres.
- Incohérence fonctionnelle front bloquante potentielle (S1):
  - Sema accepte `sin/cos/tan/exp/log/sqrt/pow` (`lib/Sema/Sema.cpp:349`),
  - MLIRGen init n’implémente que `sin` et `sqrt` (et `^` limité 0..4) (`lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp:522`),
  - donc certains programmes passent Sema et cassent en MLIRGen.
- Coordonnées:
  - Sema vérifie cohérence nom coord vs `simulation.coordinates` (`lib/Sema/Sema.cpp:65`),
  - mais `coord` MLIR porte juste un `name` string, sans liaison runtime explicite (cf. section C/D).

## B) Gap audit: Tensorium IR / Domain IR
- IR init existe (`InitExprIR`, `Metric4InitIR`, `InitialDataIR`) dans `include/tensorium/IR/IRBase.hpp:135`.
- Gap principal:
  - `InitSymbolIR` reste un symbole string (`include/tensorium/IR/IRBase.hpp:147`),
  - la distinction forte `param vs coord vs field` n’est matérialisée qu’en MLIRGen (`emitInitExpr`).
- `metric4` est aujourd’hui un "builder op" front, pas une représentation immédiatement exécutable point-wise indépendante.
- Pas de modèle IR explicite de layout mémoire de field pour sortie numérique.
- Pas d’étape IR front dédiée "evaluate init over grid points".

## C) Gap audit: Tensorium MLIR Dialect (init path)
Ops effectivement présentes dans `@tensorium_init` Schwarzschild:
- `tensorium.const`, `tensorium.param`, `tensorium.coord`,
- `tensorium.add/sub/mul/div/sin`,
- `tensorium.metric4`,
- `tensorium.decompose3p1_from_metric`,
- `tensorium.init3p1`,
- `tensorium.assign`.

État actuel par op:
- Verifier: oui pour ces ops (`lib/tensorium_mlir/Dialect/Tensorium/IR/TensoriumOps.cpp:141`, `:150`, `:261`, `:271`, `:298`, `:321`).
- Canonicalization/folding spécifique op: non (pas de patterns dédiés pour `metric4/decompose/init3p1/assign`).
- Sémantique exécutable (au sens backend/JIT): incomplète pour `param/coord/metric4/decompose/init3p1/assign`.

Constat important:
- Les passes Tensorium actuelles réécrivent surtout autour de `dt_assign`/rhs (`lib/tensorium_mlir/Dialect/Tensorium/Transforms/EinsteinLoweringPass.cpp:77`),
- pas de passe opérationnelle dédiée au trio `metric4 -> decompose3p1_from_metric -> init3p1`.

## D) Runtime contract minimal (front ABI proposal)
### ABI minimale proposée (C-like)
Option SoA (recommandée):
- entrées scalaires/globale:
  - `double M;`
  - `size_t n_points;`
- coordonnées (longueur `n_points`):
  - `const double *r;`
  - `const double *theta;`
  - `const double *phi;` (nullable si non utilisé)
- sorties:
  - `double *alpha;` (1 composante),
  - `double *gamma[9];` (cov 3x3 dense),
  - `double *gammaU[9];` (con 3x3 dense),
  - optionnel `double *g4[16];`.

### Pourquoi SoA
- simple pour vectorisation et loops backend,
- composantes tensorielles clairement adressables,
- s’aligne avec `tensorium.assign` field-oriented.

### Gaps type-system pour rendre `!tensorium.field` lowerable
`FieldType` actuel ne porte que `elementType/up/down` (`include/tensorium_mlir/Dialect/Tensorium/IR/TensoriumTypes.h:37`).
Il manque:
- base pointer / ownership,
- shape spatiale (`n_points` ou dims),
- strides/layout,
- éventuellement espace mémoire/alignment.

Sans ces infos, impossible de définir un lowering mémoire stable pour `assign/ref`.

## E) Tests front manquants avant backend
### Numeric expectations (sans backend LLVM)
- Ajouter une couche test "init evaluator" front (interprète `InitExprIR` + règle `decompose3p1_from_metric` minimale diag/beta=0),
  ou un "MLIR interpreter shim" local limité aux ops init.
- Cas minimal requis:
  - Schwarzschild point test `(M=1,r=10,theta=pi/2)` avec assertions numériques ci-dessus.

### Tests négatifs/edge cases
- `r = 2M`:
  - décider explicitement le contrat (recommandé: **autoriser** et accepter IEEE `inf/0`, pas reject front).
- `theta = 0`:
  - vérifier `g_phiphi = 0` sans NaN parasite.
- Dimension:
  - cohérence 2D axisym vs 3D spherical sur composantes partagées.

### Ce qui est déjà bien couvert
- invariants structuraux init/rhs et use-def sont déjà testés en C++:
  - `tools/Tester/UnitTests.cpp:849`, `:914`, `:1075`, `:1252`.
- garde-fous `initial_data`/diagnostics existent dans `run_test.sh`:
  - `tests/semantic/initial_data/*` (`run_test.sh:108`, `:113`).

## MLIR op -> minimal executable semantics -> backend dependency
| Op MLIR | Sémantique minimale exécutable (init-only) | Dépendance backend |
|---|---|---|
| `tensorium.const` | produire un scalaire `f64` | arith de base |
| `tensorium.param(name)` | lire un param runtime (ex: `M`) | table params/ABI |
| `tensorium.coord(name)` | lire coordonnée du point courant (`r/theta/phi`) | grid/coord provider |
| `tensorium.add/sub/mul/div` | arith scalaire `f64` point-wise | arith scalar lowering |
| `tensorium.sin/sqrt` | fonctions math `f64` point-wise | libm/runtime math |
| `tensorium.metric4(16 comps)` | construire `g_{mu,nu}` cov 4x4 au point | valeur structurée 4x4 |
| `tensorium.decompose3p1_from_metric(g)` | calculer `alpha,beta,gamma,gammaU` (au moins diag + beta=0) | algo décomposition/inversion |
| `tensorium.init3p1(a,b,g,gU)` | binding/no-op typé (ou normalisation) | convention de pipeline |
| `tensorium.assign(field, rhs)` | store dans buffer field cible | ABI mémoire field |
| `tensorium.ref(field,...)` | load depuis buffer field (utile pour vérif rhs) | ABI mémoire field |

## Recommandation: plan d’attaque front -> backend (3 phases max)
### Phase 1 — Contrat exécutable init-only (front contract first)
- Objectif:
  - figer ABI runtime minimale (`param + coord arrays + output buffers`),
  - figer sémantique opérationnelle des ops init listées ci-dessus.
- Fichiers attendus:
  - docs contrat: `docs/front_gaps_before_backend.md` (ce document),
  - types/ABI front: `include/tensorium_mlir/...` (nouveau contrat),
  - éventuellement adapter `MLIRGen` pour matérialiser interfaces.
- Risques:
  - mauvais choix de layout mémoire bloquant plus tard.
- Tests à verrouiller:
  - tests structurels init/rhs existants + nouveaux tests contrat ABI.

### Phase 2 — Rendre la chaîne init directement lowerable
- Objectif:
  - rendre `metric4/decompose/init3p1/assign` explicitement lowerables.
- Options:
  - garder les ops et écrire leur lowering dédié,
  - ou expliciter plus tôt en arith SSA (si nécessaire).
- Fichiers attendus:
  - `lib/tensorium_mlir/Target/MLIRGen/MLIRGen.cpp`,
  - passes/rewrites dédiées init.
- Risques:
  - dérive sémantique si decompose est partiellement réécrit sans invariants.
- Tests:
  - point checks numériques Schwarzschild, invariants init/rhs conservés.

### Phase 3 — Backend/JIT seulement après contrat stabilisé
- Objectif:
  - brancher lowering LLVM/JIT sur un IR init déjà stable.
- Risques:
  - confondre bugs backend et ambiguïtés front si phase 1/2 incomplètes.
- Tests:
  - réutiliser les mêmes fixtures + comparaison numérique sur points de référence.

## Action checklist (cochable)
- [ ] Spécifier ABI init-only officielle (params, coords, outputs, layout).
- [ ] Ajouter test numérique pointwise Schwarzschild (`M=1,r=10,theta=pi/2`).
- [ ] Décider et documenter le contrat horizon `r=2M` (allow IEEE vs reject).
- [ ] Harmoniser whitelist Sema vs MLIRGen pour fonctions `initial_data`.
- [ ] Définir sémantique exécutable de `decompose3p1_from_metric` (scope minimal exact).
- [ ] Ajouter tests dimension 2D/3D sur composantes partagées.
- [ ] Documenter la correspondance `!tensorium.field` -> buffers runtime (shape/strides).

