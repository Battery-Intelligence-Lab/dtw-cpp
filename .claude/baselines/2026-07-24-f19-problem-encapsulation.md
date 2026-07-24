# F19 Problem encapsulation and MATLAB writeback - 2026-07-24

## Scope and base

- Base: `926542f13e2290bb3f1020282a33d19a5257447e`
  (`docs: record F18 CUDA falsification`).
- `git status --short` produced no output.
- Subject: the frozen C++ `Problem` accessor/encapsulation gap and the four
  redundant MATLAB result-writeback calls named by PLAN F19.
- Out of scope: F20 storage-policy routing, F22 deprecation diagnostics, F26
  Python view aliasing, any F18/F40/F42 routing repair, and removal of public
  compatibility fields protected below.
- Implementation attempts: at most two. Deliberate compile/source/MATLAB
  mutation probes are tests, not implementation attempts. A failed product
  compile or runtime gate consumes an attempt; no rescue-tuning after attempt 2.

No F19-specific killed idea exists. The archive and LESSONS were searched before
choosing the field boundary.

## Binding scope adjudication

The inherited records conflict:

1. The original frozen contract at `c8c74bd` says naked configuration fields
   become private, but explicitly retains raw C++ result fields
   `clusters_ind`/`centroids_ind`, public `mip_settings`/`cuda_settings`, and
   public `init_fun`.
2. Rename-table rows 4 and 6 retain `maxIter` and `N_repetition` for one
   compatibility cycle; F22 owns their missing deprecation diagnostics.
3. M25/M37, still binding through `PLAN.md` section Binding decisions, require
   legacy raw C++ edits of `band`, `variant_params`, `missing_strategy`, and
   `distance_strategy` to be detected and reconciled. `cuda_settings` also
   participates in that semantic snapshot.
4. Commit `0672412` later described generic configuration/result privacy as F19
   debt without a dated decision superseding those promises. It also says
   `resize()` remains public, while `dtwc/Problem.hpp` already places it in the
   private section.

F19 therefore uses this explicit compatibility-preserving split:

| Fate | Exact fields | Count |
|---|---|---:|
| Private backing state with canonical reads/setters | `method`, `random_seed`, `last_iterations`, `tadpole_dc`, `lb_strategy`, `storage_policy`, `verbose`, `output_folder`, `name`, `data` | 10 |
| Retained public compatibility/contract fields | `maxIter`, `N_repetition`, `band`, `variant_params`, `missing_strategy`, `distance_strategy`, `cuda_settings`, `mip_settings`, `init_fun`, `clusters_ind`, `centroids_ind` | 11 |

This dated registration is the required PLAN decision: it corrects F19's
over-broad shorthand without weakening M25/M37, consuming F22, or silently
breaking the original raw-result promise. A future all-field privatization must
first overturn those named compatibility decisions explicitly.

The ten canonical reads are:

```text
method()
random_seed()
last_iterations()
tadpole_dc()
lb_strategy()
storage_policy()
verbose()
output_folder()
name()
data()
```

The existing setters remain authoritative. F19 adds the missing
`set_tadpole_dc`, `set_verbose`, `set_output_folder`, and compatibility-
preserving `set_name`; `last_iterations` is read-only. Python/MATLAB properties
must delegate to accessors/setters rather than address or assign private state.

## Confirmed inherited source inventory

The three named canonical C++ operations are absent:

```text
int last_iterations() const
void set_output_folder(path_t)
const std::string &name() const
```

`dtwc/Problem.hpp` exposes all 21 fields in one public block. Exact high-risk
private-target consumers found by the read-only audit include:

| Field | Product references | Test references |
|---|---:|---:|
| `method` | 9 | 20 |
| `random_seed` | 5 | 4 |
| `last_iterations` | 4 | 5 |
| `tadpole_dc` | 1 | 3 |
| `lb_strategy` | 1 | 5 |
| `storage_policy` | 1 | 3 |
| `verbose` | 13 | 55 |
| `output_folder` | 5 | 23 |
| `name` | 4 | 3 |
| `data` | 42 | 47 |

These are lexical scale indicators over usual `Problem` variable names, not a
semantic write count. The compiler fixture and final build, not this census,
judge completeness.

The MATLAB writeback inventory is exact:

| MEX command | Redundant helper call | Core writeback |
|---|---:|---:|
| `fast_pam` | `dtwc_mex.cpp:1142` | `fast_pam.cpp:532-534` |
| `fast_clara` | `dtwc_mex.cpp:1161` | `fast_clara.cpp:615-617` |
| `clarans` | `dtwc_mex.cpp:1179` | `clarans.cpp:249-251` |
| `cut_dendrogram` | `dtwc_mex.cpp:1203` | `hierarchical.cpp:247-249` |

The sole helper is `store_result_in_problem` at
`bindings/matlab/dtwc_mex.cpp:326-331`. `build_dendrogram` does not publish a
flat result, and no fifth MATLAB binding-side result writeback exists.

## Independent non-degenerate MATLAB oracle

Registered scalar L1 fixture:

```text
X =
     0  1  0  2
     1  0  2  1
     3  5  2  4
     6  2  7  3
    20 21 19 22
    22 18 25 20
    26 25 29 23
    31 28 27 35
k = 2
band = -1
device = cpu
```

The independent full-matrix dynamic program produced:

```text
F19_MATRIX=[[0, 2, 11, 15, 79, 82, 100, 118], [2, 0, 10, 14, 78, 81, 99, 117], [11, 10, 0, 8, 68, 71, 89, 107], [15, 14, 8, 0, 64, 67, 85, 103], [79, 78, 68, 64, 0, 9, 21, 39], [82, 81, 71, 67, 9, 0, 18, 36], [100, 99, 89, 85, 21, 18, 0, 22], [118, 117, 107, 103, 39, 36, 22, 0]]
F19_PAIR_COUNT=28
F19_DISTINCT_DISTANCE_VALUES=28
```

Exhaustive enumeration of all 28 two-medoid sets gives the unique optimum
zero-based set `(1,6)` at cost 87. The next-best cost is 89 and every point has
a unique nearest medoid for the optimum. Thus the MATLAB one-based optimum set
is `{2,7}`. The two clusters are exactly rows 1-4 and 5-8.

The exact registered route outputs are:

| Route/options | Labels | Medoids | Cost |
|---|---|---|---:|
| FastPAM, seed 42 | `[2 2 2 2 1 1 1 1]` | `[7 2]` | 87 |
| FastCLARA, sample 4, samples 3, seed 42 | `[2 2 2 2 1 1 1 1]` | `[7 4]` | 98 |
| CLARANS, local 3, neighbors 50, seed 42 | `[1 1 1 1 2 2 2 2]` | `[2 7]` | 87 |
| average-linkage cut | `[1 1 1 1 2 2 2 2]` | `[2 7]` | 87 |

FastCLARA's sample size is strictly smaller than N and its registered
non-optimal result distinguishes the resident sampling body from the full-
sample FastPAM delegate.

Before every target call, FastPAM with k=3 and seed 29 poisons the same Problem:

```text
labels=[2 2 2 2 1 1 1 3]
medoids=[6 2 8]
cost=53
```

The target must overwrite that state.

## Registered gates

### C1 - public-header compile contract

A standalone public-header fixture contains exactly 31 static contract
assertions:

- 10/10 private-target raw assignments are ill-formed;
- 10/10 canonical read accessors are callable on `const Problem&`;
- 11/11 retained public fields remain directly assignable/readable.

The inherited tree must fail compilation. The repaired tree must compile with
all 31 assertions present. The fixture also calls every authoritative setter;
`last_iterations` has no public setter.

### C2 - source/API guard

The permanent guard requires:

- exactly the registered 10 private and 11 retained names;
- zero private-target declarations in the `Problem` public data-member block;
- all ten backing fields in the private block;
- all ten canonical read accessors and nine setters named above;
- `resize()` remains private;
- final MATLAB source contains zero `store_result_in_problem` tokens and zero
  calls following the four algorithm commands;
- the four core writeback sites remain present.

The inherited profile must report the ten public-private violations, the three
named accessor gaps, one helper, and four calls.

### C3 - focused C++ runtime

The existing non-degenerate capped-Lloyd fixture remains the behavior oracle:

```text
data={0,40,40,46,49,51,51,51,100}
initial_medoids={0,8}
expected_medoids={1,5}
expected_labels={0,0,0,1,1,1,1,1,1}
expected_cost=96
expected_last_iterations=2
```

The new focused test additionally checks each of the ten canonical reads after
configuration through its setter, exact constructor/set-name behavior, and an
output path inside the build tree. Band: at least 30 assertions in at least
three cases, no skip text, and an `F19_PROBLEM_API ... skips=0` marker.

### M1 - MATLAB writeback mutation proof

Six source profiles execute under both installed releases:

| Profile | Target routes per release |
|---|---:|
| inherited baseline | 4 |
| delete FastPAM helper call only | 1 |
| delete FastCLARA helper call only | 1 |
| delete CLARANS helper call only | 1 |
| delete cut-dendrogram helper call only | 1 |
| delete all four calls and helper | 4 |

Band: exactly 24 target-route executions. Every execution asserts exact
`int32` class, row shape, values, returned-versus-stored labels/medoids, k=2,
and the registered cost. There are at least 192 exact vector identity
assertions before scalar/class/shape checks. Each single deletion and the
composite deletion must survive digit-identically.

Every profile records source SHA-256, MEX SHA-256, a compile line proving
`dtwc_mex.cpp` rebuilt, and exact `which('dtwc_mex','-all')`. MATLAB starts
with `restoredefaultpath`, adds `bindings/matlab` first and the fresh binary
directory last, then clears MEX state. The source is restored byte-for-byte
between deliberate mutants; the final product source is the composite deletion.

### B1 - bindings and full gates

- Fresh ordinary R2024b MEX builds; the isolated F19 oracle passes on R2024b
  and R2025b. Known retained F18 red cases are not counted as F19 failures.
- Fresh Python extension exposes the same property values through the new C++
  accessors/setters; full current pytest floor remains at least 1010 passed /
  12 skipped over 1022 collected.
- Canonical and llfio-OFF C++ suites: zero failures, exactly their registered
  capability skips, and at least the inherited 120 tests discovered.
- Documentation generation/contract and record-hygiene checks pass.
- Known F39 supply-chain gate remains recorded by its exact observed inventory;
  F19 must not falsely claim to close it.

### Mutations and attempts

Besides the five MATLAB deletion profiles, the permanent source/compile gate
must reject at least these 12 mutations:

1. expose any one of the ten backing fields publicly;
2. remove `last_iterations()`;
3. make `last_iterations()` mutable;
4. remove `set_output_folder`;
5. make `name()` return a mutable reference;
6. remove any one retained compatibility field;
7. move `resize()` public;
8. restore any one of the four MATLAB helper calls;
9. restore the helper under another name but the same three assignments;
10. bind Python `output_folder` by direct assignment;
11. bind MATLAB verbose by direct assignment;
12. replace a core writeback with binding-only writeback.

The claim most likely to be wrong is that current F19 text intended privacy for
all result/configuration fields. The original frozen contract and the still-
binding M25/M37/F22 decisions support only the registered 10/11 split; an
all-field interpretation would require a separate explicit compatibility
break.
