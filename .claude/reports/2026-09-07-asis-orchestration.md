# AS-IS map — orchestration layer (`Problem` / `Data` / `DataLoader` / api / checkpoint / env / scores)

Repo `C:\D\git\dtw-cpp`, branch `Claude`, HEAD `a31956e`. Read-only pass; every line of the
20 assigned files was read. Evidence tags: `[confirmed path:line]` = read directly in the tree at
HEAD; `[inferred]` = reasoned, with the confirming test named; `[not established]` = unknown.

Scope note: this is a MAP plus a PROBLEM INVENTORY. Each problem carries at most a one-line
candidate action. Structures that look odd but are deliberate are tagged **DELIBERATE** in §5.

---

## 1. Module map

| File | Lines | Responsibility (one line) | Public entry points | Notable includes |
|---|---|---|---|---|
| `dtwc/Problem.hpp` | 704 | The god-object: data ownership, DTW binding, distance-matrix cache + fingerprints, clustering state, IO, settings structs | `class Problem`; `CUDASettings`, `MIPSettings`, `DistanceMatrixStrategy`, `validate_*` free fns | `Data.hpp`, `DataLoader.hpp`, `initialisation.hpp`, `checkpoint.hpp`, `core/{dtw_options,storage,distance_matrix,mmap_distance_matrix}.hpp`, `<variant>`, `<functional>`, `<atomic>` |
| `dtwc/Problem.cpp` | 1460 | Fingerprints, cache-identity/preflight machinery, DTW rebinding, matrix fill dispatch, Lloyd k-medoids, MIP dispatch | `Problem::*` (non-IO) | `mip.hpp`, `parallelisation.hpp`, `core/{dtw_dispatch,distance_semantics,variant_validation,pruned_distance_matrix,sha256}.hpp`, `algorithms/tadpole.hpp` |
| `dtwc/Problem_IO.cpp` | 274 | CSV writers for medoids/clusters/silhouettes/matrix + the CSV matrix reader | `Problem::{writeMedoids,print_clusters,write_clusters,write_silhouettes,write_medoid_members,write_distance_matrix,writeBestRep,read_distance_matrix}` | `core/matrix_io.hpp`, `scores.hpp`, `types/Range.hpp` |
| `dtwc/Data.hpp` | 207 | Series container with four modes (heap-f64, heap-f32, view-spans, metadata-only) | `struct Data`, `Data::metadata_only` | `settings.hpp`, `core/storage.hpp`, `<span>` |
| `dtwc/DataLoader.hpp` | 594 | Builder-pattern loader + the single storage-routing primitive `route_series_storage` | `class DataLoader`, `struct LoadedData`, `detail::{available_ram_bytes,series_footprint_bytes,series_storage_threshold,choose_storage,route_series_storage}` | `fileOperations.hpp`, `env.hpp`, `core/mmap_data_store.hpp`, `<atomic>`, `<random>` |
| `dtwc/system_memory.cpp` | 75 | Platform free-RAM query, isolated in a .cpp so `<windows.h>` never reaches the umbrella header | `detail::available_ram_bytes()` | `DataLoader.hpp`, `<windows.h>`/`<unistd.h>`/`<mach/*>` |
| `dtwc/checkpoint.hpp` | 143 | Declares `CheckpointOptions` + the dense-generation and binary-result checkpoint APIs | `CheckpointOptions`, `save_checkpoint`, `load_checkpoint`, `save_binary_checkpoint`, `load_binary_checkpoint` | `core/clustering_result.hpp`, `core/dtw_options.hpp`, `error.hpp` |
| `dtwc/checkpoint.cpp` | 974 | Dense checkpoint v2 (immutable generations, SHA-256 manifest, atomic CURRENT) + little-endian binary result format | the four functions above | `Problem.hpp` (friend access to `Problem::distMat`), `core/sha256.hpp`, `<bit>`, `<charconv>` |
| `dtwc/initialisation.hpp` | 26 | Declares the four medoid initialisers | `init::{random,Kmeanspp,random_seeded,Kmeanspp_seeded}` | `<cstdint>` only (forward-declares `Problem`) |
| `dtwc/initialisation.cpp` | 217 | Shuffle- and D-sampling initialisers, templated over an injected RNG strategy | as above | `core/portable_random.hpp`, `core/distance_sampling_weights.hpp`, `Problem.hpp` |
| `dtwc/scores.hpp` | 63 | Declares 7 canonical validity indices + 5 deprecated camelCase shims | `scores::{silhouette,davies_bouldin,dunn,inertia,calinski_harabasz,adjusted_rand,normalized_mutual_info}` | `<vector>` only (forward-declares `Problem`) |
| `dtwc/scores.cpp` | 498 | Implementations; realised-partition guards shared through `cluster_counts_checked` | as above | `Problem.hpp`, `parallelisation.hpp` |
| `dtwc/env.hpp` | 176 | Process-wide device + thread policy; the no-silent-fallback contract | `enum Device`, `class Env`, `env()`, `to_string(Device)`, `warn_if_single_threaded()`, `detail::{sequential_cause,sequential_warning_text}` | `error.hpp`, `<functional>`, `<filesystem>` |
| `dtwc/env.cpp` | 376 | Device-name parsing, `.env` parsing, ssh auth probe, once-per-process sequential warning | as above | `<omp.h>`, `<mutex>`, `<sys/wait.h>` |
| `dtwc/api.hpp` | 112 | Tier-1 surface: `load` → `Dataset` → `cluster` → `Result` | `Dataset`, `Result`, `load` (×2 + 2 deleted poison overloads), `cluster`, `device` (×2) | `Data.hpp`, `<variant>`, `<memory>` |
| `dtwc/api.cpp` | 412 | Tier-1 implementation: string method normalisation, device configuration, algorithm routing, `Result::save` | as above | `Problem.hpp`, `DataLoader.hpp`, `algorithms/*`, `detail/tier1_method_resolution.hpp`, `env.hpp`, `scores.hpp` |
| `dtwc/test_api.hpp` | 276 | **Production** self-introspection probes (`parallelisation()`, `gpu()`), header-only, bound into Python and MATLAB | `dtwc::test::{ParallelReport,GpuReport,parallelisation,gpu}` | `dtwc.hpp` (the whole umbrella), `<omp.h>` |
| `dtwc/detail/tier1_method_resolution.hpp` | 54 | Two `constexpr` pure functions: `auto`-method resolution and GPU storage pinning | `Tier1ExecutionTarget`, `resolve_tier1_method`, `tier1_storage_policy` | `core/storage.hpp` |
| `dtwc/dtwc.hpp` | 61 | Fat umbrella header — 38 unconditional includes plus MPI/CUDA/Metal blocks | — | everything |
| `dtwc/main.cpp` | 41 | Example driver (`dtwc_main`) | `main()` | `dtwc.hpp` |

---

## 2. `Problem` anatomy — responsibility clusters

Declared members at HEAD: **25 private data**, **12 public data**, **~32 private member functions**,
**~105 public member functions** (22 of them `[[deprecated]]` shims; several are const/non-const
overload pairs) ≈ **174 declared members** `[confirmed dtwc/Problem.hpp:122-701]`.

### C1 — Data ownership (4 members)

| Member | Line |
|---|---|
| `Data data_` | `Problem.hpp:198` |
| `std::unique_ptr<LoadedData> series_storage_owner_` | `Problem.hpp:197` |
| `void adopt_loaded_data(LoadedData)` | `Problem.hpp:268-294` |
| `bool has_mmap_series_storage() const` | `Problem.hpp:296-300` |

**Invariants.** (a) When the storage route is mmap, `data_` is a *view* whose spans and
`string_view` names point into `*series_storage_owner_`; `adopt_loaded_data` asserts pointer
identity (`view.name(i).data() != owner->names[i].data()` → `std::logic_error`)
`[confirmed dtwc/Problem.hpp:280-286]`. (b) On the heap route `series_storage_owner_` is reset to
null `[confirmed dtwc/Problem.hpp:292-293]`. (c) `data_.validate_ndim()` has run before adoption.

**Touches.** C2 (rebinding reads `data_.ndim` and the series), C3 (identity hashes every value),
C6 (writers read names), and DataLoader (`route_series_storage` produces the `LoadedData`).

### C2 — DTW binding / dispatch (11 members)

| Member | Line |
|---|---|
| `mutable dtw_fn_t dtw_fn_` | `Problem.hpp:137` |
| `mutable dtw_fn_f32_t dtw_fn_f32_` | `Problem.hpp:138` |
| `mutable const Problem *dtw_binding_owner_` | `Problem.hpp:139` |
| `mutable std::unordered_map<size_t,std::vector<data_t>> wdtw_weights_cache_` | `Problem.hpp:140` |
| `void rebind_dtw_fn() const` | `Problem.hpp:213`, `Problem.cpp:298-320` |
| `void refresh_variant_caches() const` | `Problem.hpp:214`, `Problem.cpp:262-293` |
| `const dtw_fn_f32_t &validated_dtw_function_f32() const` | `Problem.hpp:234`, `Problem.cpp:436-445` |
| `void repair_dtw_binding_after_relocation()` | `Problem.hpp:235`, `Problem.cpp:447-460` |
| `void ensure_dtw_function_configuration_current()` | `Problem.hpp:243`, `Problem.cpp:494-506` |
| `void validate_dtw_function_configuration() const` | `Problem.hpp:244`, `Problem.cpp:508-526` |
| public `dtw_function()` ×2, `dtw_function_f32()` ×2, `wdtw_weights_cache()` | `Problem.hpp:558-589` |

**Invariants.** (a) The closures capture `*this` and read `band`, `variant_params`,
`missing_strategy`, `data_.ndim`, `wdtw_weights_cache_` *at call time*
`[confirmed dtwc/Problem.cpp:299-304]`. (b) `dtw_binding_owner_ == this` is the "closures point at
me" predicate; a defaulted move breaks it and every public gateway repairs it before use
`[confirmed dtwc/Problem.cpp:447-460, 516-519]`. (c) `wdtw_weights_cache_` is populated **serially
and exhaustively** before any parallel fill; nothing inserts into it afterwards
`[confirmed dtwc/Problem.cpp:271-274]`. (d) `dtw_fn_f32_` is empty when the active variant is not
f32-representable; `validated_dtw_function_f32()` turns that into a `logic_error`
`[confirmed dtwc/Problem.cpp:436-445]`.

**Touches.** C1 (`ndim`, series), C3 (`rebind_dtw_fn` writes the dense configuration snapshot at
`Problem.cpp:317-318`), C7 (reads `band` / `variant_params` / `missing_strategy`).

### C3 — Distance-matrix cache + configuration fingerprints (17 members)

| Member | Line |
|---|---|
| `distMat_t distMat` (`variant<Dense,Mmap>`) | `Problem.hpp:135` |
| `struct DistanceCacheConfiguration` | `Problem.hpp:143-151` |
| `struct DistanceCacheIdentity` | `Problem.hpp:152-159` |
| `class RelaxedFlag` | `Problem.hpp:165-176` |
| `DistanceCacheIdentity mmap_cache_identity_` | `Problem.hpp:178` |
| `bool mmap_cache_identity_bound_` | `Problem.hpp:179` |
| `mutable RelaxedFlag mmap_cache_data_validated_` | `Problem.hpp:180` |
| `mutable DistanceCacheConfiguration dense_cache_configuration_` | `Problem.hpp:181` |
| `mutable bool dense_cache_configuration_bound_` | `Problem.hpp:182` |
| `visit_distmat` ×2 | `Problem.hpp:202-211` |
| `distance_cache_configuration_fingerprint`, `distance_cache_configuration`, `distance_cache_configuration_matches`, `dense_cache_configuration_is_current`, `dense_cache_metric` | `Problem.hpp:215-224`; `Problem.cpp:342-406` |
| `ensure_dense_cache_configuration_current`, `..._preflighted`, `validate_dense_cache_configuration` | `Problem.hpp:236-242`; `Problem.cpp:462-492` |
| `distance_cache_identity`, `validate_mmap_cache_identity`, `clear_mmap_cache_identity` | `Problem.hpp:245-248`; `Problem.cpp:528-635` |
| public `use_mmap_distance_matrix`, `distance_checkpoint_identity`, `distance_matrix()` ×2, `dense_distance_matrix()` ×2, `max_distance`, `is_distance_matrix_filled` | `Problem.hpp:542-642` |
| `resize()` | `Problem.hpp:250`; `Problem.cpp:120-124` |

**Invariants.** (a) Two-tier identity: the **full** SHA-256 (every IEEE value, lengths, order,
precision, ndim, plus the configuration digest) is verified at bind and once more at first use;
after that only the **fixed-size** `DistanceCacheConfiguration` snapshot is compared, so warm
lookups stay O(1) `[confirmed dtwc/Problem.cpp:552-576, 618-634]`. (b) Series *names* are
deliberately excluded from the identity `[confirmed dtwc/Problem.cpp:557-560]`. (c) A dense matrix
whose snapshot no longer matches the live configuration is not readable: const paths throw, mutable
paths refresh `[confirmed dtwc/Problem.cpp:462-492]`. (d) A non-L1 mmap cache is external-fill-only
`[confirmed dtwc/Problem.cpp:717-723, 865-870]`. (e) `resize()` sizes **`clusters_ind` /
`centroids_ind`**, not the matrix `[confirmed dtwc/Problem.cpp:120-124]`.

**Touches.** C1 (identity hashes the data), C2 (`rebind_dtw_fn` writes the snapshot), C5 (mid-fill
autosave tags the generation with `dense_cache_metric()`), C7 (every configuration field).

### C4 — Clustering state and method dispatch (12 members)

`Nc` (`Problem.hpp:134`), `mipSolver` (`:136`), `method_` (`:184`), `random_seed_` (`:185`),
`last_iterations_` (`:186`), `tadpole_dc_` (`:187`), `lb_strategy_` (`:188`), public
`clusters_ind` / `centroids_ind` (`:321-322`), public `init_fun` (`:319`),
`cluster_by_kmedoids_lloyd_impl` / `cluster_by_kMedoidsLloyd_single` / `init_with_seed`
(`:259-262`), plus public `cluster`, `cluster_by_mip`, `cluster_by_kmedoids_lloyd`,
`cluster_and_process`, `assign_clusters`, `calculate_medoids`, `find_total_cost`, `init`,
`set_clusters`, `set_n_clusters`, `centroid_of`, `labels`, `medoids` (`:404-700`).

**Invariants.** (a) `clusters_ind.size() == size()` and `centroids_ind.size() == Nc` — maintained by
`resize()`, called from `set_n_clusters` only `[confirmed dtwc/Problem.cpp:132-136]`.
(b) `set_clusters` rejects a candidate whose size ≠ `Nc` `[confirmed dtwc/Problem.cpp:186-192]`.
(c) `cluster()` is side-effect free by contract; artefacts belong to `cluster_and_process()` via the
RAII `ArtifactScope` `[confirmed dtwc/Problem.cpp:1149-1156]`. (d) `init_with_seed` recognises
exactly the two library initialisers by function-pointer target and otherwise falls back to the
legacy unseeded call `[confirmed dtwc/Problem.cpp:1281-1298]`.

**Touches.** C3 (every cost/assignment goes through `dist_by_ind`), C6 (artefact writers),
`initialisation.cpp`, and every `algorithms/*` file, which write `clusters_ind` / `centroids_ind`
directly `[confirmed dtwc/algorithms/fast_pam.cpp:537-539, one_batch_pam.cpp:396-398,
fast_clara.cpp:568-570, hierarchical.cpp:279-281]`.

### C5 — Checkpoint identity and options (4 members)

Public `CheckpointOptions checkpoint` (`Problem.hpp:317`), `validate_checkpoint_settings() const`
(`:247`, `Problem.cpp:757-774`), `distance_checkpoint_identity(metric)` (`:629`,
`Problem.cpp:579-587`), and `friend bool load_checkpoint(Problem&, ...)` (`:254-255`).

**Invariants.** (a) `enabled` requires `save_interval >= 1`, a non-empty directory and **dense**
storage; all three are rejected before any distance is computed
`[confirmed dtwc/Problem.cpp:757-774]`. (b) Saves happen on the calling thread after `run_openmp`
has joined a whole row block, so no partially-written row is ever observed
`[confirmed dtwc/Problem.cpp:832-844]`. (c) The generation tag is `dense_cache_metric()`, never a
literal `[confirmed dtwc/Problem.cpp:841-843, 1108-1109]`.

### C6 — Outputs / IO (11 members)

Private `writeBestRep`, `writeMedoids`, `distanceInClusters` (`Problem.hpp:264-266`); public
`print_clusters`, `write_clusters`, `write_silhouettes`, `write_medoid_members`,
`write_distance_matrix` ×2, `print_distance_matrix`, `read_distance_matrix`, plus
`output_folder_` / `name_` / `persist_run_artifacts_` (`Problem.hpp:194-196`).

**Invariants.** (a) Every writer creates its parent directory and checks open **and** close
`[confirmed dtwc/Problem_IO.cpp:34-62]` — with one gap (§8 D6). (b) `write_silhouettes` swallows
only `UndefinedScore` `[confirmed dtwc/Problem_IO.cpp:161-166]`. (c) `read_distance_matrix`
propagates `[confirmed dtwc/Problem_IO.cpp:261-272]`.

### C7 — Settings / configuration surface (17 members)

Public data: `band` (`:307`), `variant_params` (`:312`), `missing_strategy` (`:313`),
`distance_strategy` (`:314`), `cuda_settings` (`:315`), `mip_settings` (`:316`), `checkpoint`
(`:317`), deprecated `maxIter` (`:304`) and `N_repetition` (`:306`). Private: `storage_policy_`
(`:189`), `ram_limit_bytes_` (`:190`), `verbose_` (`:191`), plus the encapsulated eleven read
accessors (`:426-436`) and their setters (`:438-507`).

**Invariants.** (a) *Every* setter that changes distance semantics runs `preflight_*` **before**
mutating, then compares-and-returns, then `refresh_distance_matrix()`
`[confirmed dtwc/Problem.hpp:443-497, 509-536]`. (b) Setters that do **not** change distances
(`set_lb_strategy`, `set_storage_policy`, `set_ram_limit`, `set_random_seed`, `set_verbose`,
`set_output_folder`, `set_name`, `set_method`, `set_tadpole_dc`) deliberately do not invalidate
`[confirmed dtwc/Problem.hpp:472-507]`. (c) `storage_policy_` governs the **next** owning
`set_data` and is non-retroactive `[confirmed dtwc/Problem.hpp:483-484]`.
(d) `set_max_iter` / `set_n_repetitions` write straight into the deprecated public fields with **no
validation** `[confirmed dtwc/Problem.cpp:153-171]`.

### C8 — Lifetime / friendship (7 members)

`Problem()`, `Problem(name)`, `Problem(name, DataLoader&)` (`:339-350`), deleted copy (`:358-359`),
move ctor + move assign (`:360-361`, defaulted at `Problem.cpp:149-151`), and three friends:
`ProblemStoragePolicyTestAccess`, `load_checkpoint`, `MIP_clustering_byBenders` (`:253-256`).

---

## 3. State machine

```
                       Problem()            Problem(name)          Problem(name, DataLoader&)
                       hpp:339              hpp:340-343            hpp:344-350
                          |                     |                       |
                          +---- rebind_dtw_fn() +                       | adopt_loaded_data(loader.load_stored())
                                                                        | refresh_distance_matrix()  -> rebind
                                                                        v
   +-------------------------------------------------------------------------------------------+
   |  CONFIGURED (data_ + dtw_fn_ + dense_cache_configuration_ snapshot all agree)              |
   +-------------------------------------------------------------------------------------------+
      |            |                |                 |                   |
      | set_data   | set_view_data  | set_band /      | set_lb_strategy / | RAW PUBLIC WRITE
      | hpp:509    | hpp:526        | set_variant /   | set_storage_policy| (band=..., variant_params.x=...)
      |            |                | set_missing_... | set_ram_limit ... | main.cpp:30, MEX 1217/1576
      v            v                | set_distance_...| set_random_seed   v
  validate_precision                | set_cuda_settings                (no invalidation yet)
  validate_ndim                     v                 v                   |
  preflight_distance_semantics   preflight (BEFORE mutation)   NO invalidation
  route_series_storage           compare-and-return (no-op if equal)      |
  adopt_loaded_data              refresh_distance_matrix()                |
  refresh_distance_matrix()        cpp:241-260:                           |
  [set_view_data ALSO resize()]      - preflight again                    |
                                     - Mmap  -> replace with empty Dense  |
                                               + clear_mmap_cache_identity|
                                     - Dense -> m.resize(0) (release;     |
                                               alloc deferred to fill)    |
                                     - rebind_dtw_fn()  cpp:298-320:      |
                                         preflight, refresh_variant_caches|
                                         resolve f64 + f32, write         |
                                         dense_cache_configuration_,      |
                                         dtw_binding_owner_ = this        |
                                              |                           |
                                              v                           v
   +-------------------------------------------------------------------------------------------+
   |  BOUND, matrix size 0 (allocation deferred)                                                |
   +-------------------------------------------------------------------------------------------+
      |                        |                                    |
      | use_mmap_distance_matrix(path, metric)  cpp:637-663         | first dist_by_ind(i,j)  cpp:678-730
      |   validate_metric_type, preflight,                          |   preflight
      |   ensure_dtw_function_configuration_current                 |   validate_mmap_cache_identity  <-- preflights AGAIN
      |   distance_cache_identity(metric)  (FULL data hash)         |   ensure_dense_..._preflighted
      |   open(path, identity) OR create(path, N, identity)         |   omp critical(distByInd_init): m.resize(N);
      |   mmap_cache_identity_ = identity; bound_ = true;           |     rebind_dtw_fn()   <-- mutates shared state
      |   mmap_cache_data_validated_ = false                        |   compute + m.set(i,j,d)
      v                                                             v
   +-------------------------------------------------------------------------------------------+
   |  fill_distance_matrix()   cpp:856-1113                                                     |
   |    validate_checkpoint_settings -> preflight -> validate_lower_bound_strategy               |
   |    -> validate_mmap_cache_identity (first-use FULL data hash, sets RelaxedFlag)             |
   |    -> ensure_dense_cache_configuration_current -> early-return if already filled            |
   |    -> reject non-L1 mmap -> allocate Dense if size != N -> rebind_dtw_fn()                  |
   |    -> SERIAL missing pre-scan (Error: any NaN; Interpolate: all-NaN)   cpp:891-918          |
   |    -> resolve Auto: Pruned iff pruned_strategy_applicable, else BruteForce  cpp:920-929     |
   |    -> DeviceError if mmap series + CUDA/Metal        cpp:931-940                            |
   |    -> downgrade Pruned -> BruteForce for (missing!=Error | mmap storage | checkpointing)    |
   |    -> reject Pruned + f32 (BELOW the downgrades, deliberately)   cpp:984-990                |
   |    -> switch: Pruned | CUDA | Metal | BruteForce ; Auto == logic_error                      |
   |    -> if checkpoint.enabled && !BruteForce: one save_checkpoint  cpp:1108-1109              |
   +-------------------------------------------------------------------------------------------+
                                    |
                                    v  (BruteForce path, cpp:781-845)
                  lock-free row fill: each worker owns a disjoint row i, writes (i, j>i)
                  checkpoint.enabled -> blocks of save_interval rows; save AFTER join
                                    |
                                    v
   +-------------------------------------------------------------------------------------------+
   |  FILLED  -> cluster()  cpp:1118-1141   switch(method_): Kmedoids | MIP | LRCore | TADPole   |
   |               Kmedoids -> cluster_by_kmedoids_lloyd_impl  cpp:1311-1367                     |
   |                 fill_distance_matrix(); per repetition: init_with_seed -> single()          |
   |                 single(): assign_clusters -> distanceInClusters -> calculate_medoids        |
   |                 best repetition wins; last_iterations_ = best_iterations                    |
   |             -> cluster_and_process()  cpp:1147-1162  (ArtifactScope sets persist_run_       |
   |                artifacts_, then print/write_distance_matrix/write_clusters/write_silhouettes|
   +-------------------------------------------------------------------------------------------+
                                    |
      +-----------------------------+------------------------------+
      | scores::* (all call prob.fill_distance_matrix() first)      | save_checkpoint(const Problem&,dir,metric)
      | scores.cpp:124,200,264,301,340                              |   prob.dense_distance_matrix()  (const, validates)
      v                                                             |   prob.distance_checkpoint_identity(metric) FULL HASH
   Result / CSV artefacts                                           |   stream generation, atomic CURRENT replace, prune
                                                                    v
                                                    load_checkpoint(Problem&,dir,metric) cpp:595-635
                                                      friend access to prob.distMat (checkpoint.cpp:601)
                                                      validate_dense_cache_configuration
                                                      identity + manifest + payload + symmetry checks
                                                      publish with ONE nothrow move
```

**Move.** `Problem(Problem&&)` and `operator=(Problem&&)` are `= default`
`[confirmed dtwc/Problem.cpp:149-151]`. A move transfers `dtw_fn_` intact, so the closures still
name the **source** address. Recovery is lazy, at the first gateway:
`repair_dtw_binding_after_relocation()` rebinds if the snapshot is current, otherwise falls all the
way back to `refresh_distance_matrix()` `[confirmed dtwc/Problem.cpp:447-460]`; the const path does
the same but throws on true drift `[confirmed dtwc/Problem.cpp:508-526]`.

**What invalidates what.**

| Trigger | Dense matrix | Mmap matrix | `dtw_fn_` / `dtw_fn_f32_` | `wdtw_weights_cache_` | `clusters_ind` / `centroids_ind` |
|---|---|---|---|---|---|
| `set_data` | resize(0) | detached + identity cleared | rebound | rebuilt | **untouched** (see D3) |
| `set_view_data` | resize(0) then `resize()` | detached | rebound | rebuilt | resized |
| `set_band` / `set_variant` / `set_missing_strategy` / `set_distance_strategy` / `set_cuda_settings` | resize(0) | detached | rebound | rebuilt | untouched |
| raw write to `band` / `variant_params` / … | **deferred**: detected by the snapshot at the next gateway, then treated as a setter | same | same | same | untouched |
| `set_n_clusters` | untouched | untouched | untouched | untouched | resized |
| move | preserved if snapshot current | preserved | repaired at first gateway | preserved | preserved |

---

## 4. Data and control flow

### 4.1 Tier-1: `load` → `Dataset` → `cluster` → `Result`

`load()` stores a `variant<path, vector<vector<double>>>` and never opens a file
`[confirmed dtwc/api.cpp:185-199]`. Two `= delete` overloads poison the pre-2.0 3-argument shape so
`load(p, 0, ',')` is a compile error rather than `skip_rows = 44`
`[confirmed dtwc/api.hpp:66-69]`.

`cluster()` `[confirmed dtwc/api.cpp:322-410]`, in order: validate `k` / `max_iter` →
`normalize_method` (lowercase, `-`→`_`, `hclust`→`hierarchical`, membership check) → resolve the
device through a **local** `Env` when a per-call override is given, so the process default is never
mutated (`:334-341`) → HPC rejected with `DeviceError` (`:343-348`) →
`make_shared<Problem>(name)` → `set_storage_policy(tier1_storage_policy(target))` **before**
`set_data`, because GPU cannot read mmap series (`:355-357`) →
`set_data(dataset.materialize_local())` → `set_band` → `set_max_iter` → `configure_device` →
`resolve_tier1_method` (`auto` → `pam` if N ≤ 5000 else `clara`; GPU always `pam`; HPC keeps
`auto`) `[confirmed dtwc/detail/tier1_method_resolution.hpp:29-36]` → matrix-free methods
(`onebatch` / `clara` / `tadpole`) skip the fill and are rejected on GPU (`:369-377`) → algorithm
call.

`Result` holds `shared_ptr<Problem>` and reads through it: `labels()` / `medoids()` return
`problem_->labels()` / `medoids()` `[confirmed dtwc/api.cpp:213-214]`. This works **only** because
every algorithm writes back into the Problem's public fields
`[confirmed dtwc/algorithms/fast_pam.cpp:537-539]` — an undocumented cross-module invariant
(§8 D2).

`Dataset::materialize_local()` `[confirmed dtwc/api.cpp:151-183]`: the path branch builds a
`DataLoader` and calls `load_local()` (explicit, so a Tier-1 `device="cpu"` override never has to
mutate the global Env `[confirmed dtwc/DataLoader.hpp:427-435]`). The in-memory branch does
`auto series = std::get<series_type>(source_);` — a **full copy of the dataset**
`[confirmed dtwc/api.cpp:166]`, forced by `cluster(const Dataset&)`.

### 4.2 DataLoader routes

- **heap** — `load_heap()` (`DataLoader.hpp:515-525`) calls `load_folder` / `load_batch_file` and
  bumps the relaxed instrumentation counter once per load.
- **metadata-only (`device == HPC`)** — `load()` branches on `dtwc::env().device()` (`:420-425`);
  `load_metadata_file` / `load_metadata_folder` count fields without storing them and return
  `Data::metadata_only` (`:531-590`). `series()` then throws by design
  `[confirmed dtwc/Data.hpp:199-205]`.
- **storage-policy-aware** — `load_stored()` (`:464-478`) is HPC-aware, then delegates to the single
  routing primitive.
- **f32** — no loader produces f32; `Data`'s f32 constructor is reached only from the Python
  binding `[confirmed python/src/_dtwcpp_core.cpp:774]`.

`detail::route_series_storage` (`DataLoader.hpp:165-258`) is the **single** primitive shared by
`DataLoader::load_stored` and `Problem::set_data`: metadata-only short-circuits; otherwise it
computes `series_footprint_bytes` (O(N)) and `available_ram_bytes()` (a syscall), decides
`want_mmap`, refuses (or loudly warns for `Auto`) on f32, creates a `MmapDataStore`, then builds
`out.names` (owning) + spans + name views and returns the `LoadedData` bundle. `LoadedData` is
move-only when mmap is compiled in `[confirmed dtwc/DataLoader.hpp:70-80]`.

Threshold policy: `override` if set, else `available / 2`, else `SIZE_MAX` — i.e. **unknown free
RAM never spills** `[confirmed dtwc/DataLoader.hpp:107-129]`.

### 4.3 Checkpoint save/load and identity

`save_checkpoint(const Problem&, path, metric)` `[confirmed dtwc/checkpoint.cpp:453-592]`: complete
validation before any filesystem effect (dense-only, N > 0, dimension match, packed-count overflow,
every stored value finite-or-NaN), then `prob.distance_checkpoint_identity(metric)` → a **full
re-hash of every series value** `[confirmed dtwc/Problem.cpp:561-575]`, then stream
`generations/<id>/distances.csv` (full N×N; empty field = uncomputed) hashing as it goes, write the
7-key `metadata.txt`, then a temp file + `MoveFileExW` / `rename` to publish `CURRENT`
(`:433-448, 572-577`), then best-effort prune of superseded generations (`:578-591`). A
`GenerationCleanup` destructor removes everything if publication did not happen (`:397-416`).

`load_checkpoint(Problem&, path, metric)` `[confirmed dtwc/checkpoint.cpp:595-635]`: reaches
`prob.distMat` through `std::get_if` **as a friend** (`:601`) so no mutable accessor runs and no
Problem state changes before the final `*destination = std::move(candidate)`, whose no-throw
guarantee is `static_assert`-ed (`:627-630`).

### 4.4 How `dist_by_ind` / `series(i)` / `p_vec(i)` avoid copies — the owner's example

This is the load-bearing detail. Verbatim chain:

1. `Problem::series(size_t i) const { return data_.series(i); }`
   `[confirmed dtwc/Problem.hpp:396]` — returns `std::span<const data_t>`, **by value, no
   allocation**.
2. `Data::series(size_t i) const` `[confirmed dtwc/Data.hpp:53-59]` branches three ways and in every
   branch returns a span over storage it does not own:
   ```cpp
   if (is_metadata_only_) throw_not_resident();
   if (is_f32()) throw_wrong_precision("series", "Float32", "series_f32");
   if (is_view_) return p_spans_[i];
   return std::span<const data_t>(p_vec[i]);
   ```
   The heap branch constructs the span from `p_vec[i]` in place; the mmap/CLARA branch returns the
   stored span, which points into the `MmapDataStore` or the parent `Data`.
3. `Problem::dist_by_ind(int i, int j)` `[confirmed dtwc/Problem.cpp:725-729]`:
   ```cpp
   const double d = data_.is_f32()
                      ? validated_dtw_function_f32()(data_.series_f32(i), data_.series_f32(j))
                      : dtw_fn_(series(i), series(j));
   visit_distmat([&](auto &m) { m.set(i, j, d); });
   ```
   The DTW callable's parameter type is `std::span<const data_t>`
   `[confirmed dtwc/Problem.hpp:130]`, so the two series cross the `std::function` boundary as
   16-byte (ptr, len) descriptors. Nothing is copied at any depth.
4. Binding happens **once per Problem**, not per call: `rebind_dtw_fn()` is invoked from the
   constructors and from `refresh_distance_matrix()` only
   `[confirmed dtwc/Problem.hpp:339-350; dtwc/Problem.cpp:259, 883]`; the resolved closure reads the
   live configuration through `*this` at call time `[confirmed dtwc/Problem.cpp:299-304]`.
5. The row fill hoists the outer span once per row and reuses it across the inner loop:
   `const auto si = series(i); for (j...) m.set(i, j, dtw_fn_(si, series(j)));`
   `[confirmed dtwc/Problem.cpp:808-825]`.
6. `Problem::p_vec(size_t i)` `[confirmed dtwc/Problem.hpp:384-393]` returns `auto&` — a reference
   to `data_.p_vec[i]`, again no copy — guarded by `assert(!data_.is_view())` because a view has no
   `p_vec` to reference.
7. The fill is lock-free: "each worker owns a disjoint row", and `DenseDistanceMatrix` documents "no
   locking, no atomics" `[confirmed dtwc/Problem.cpp:805-807; dtwc/core/distance_matrix.hpp:38-41]`.

---

## 5. Performance-critical structures — DO-NOT-BREAK list

| # | Structure | Evidence | What a naive cleanup costs |
|---|---|---|---|
| P1 | `dtw_fn_` bound **once per Problem** as a `std::function`, taking `span` | `dtwc/Problem.hpp:130`; `dtwc/Problem.cpp:298-320`; `.claude/design.md:16` "`resolve_dtw_fn` binds once in `Problem` … `std::function` dispatch cost is negligible next to DTW runtime" | A per-call `switch` reinstates the ~130-line nested switch that silently bound `dtw_fn_f32_` to Standard DTW regardless of variant (`Problem.cpp:306-309`) — a silent wrong answer, not a slowdown. **DELIBERATE.** |
| P2 | Zero-copy `span` accessors with a branch on the precision/view/metadata discriminator | `dtwc/Data.hpp:50-76`; `.claude/LESSONS.md:416-420` "An accessor that does not branch on the same discriminator as its `size()` is undefined behaviour waiting for a caller … the audit's own perf note demanded a branch, not a lock or a virtual" | Removing the "redundant" branch as a micro-optimisation indexes the empty `p_vec` / `p_spans_` on f32 data. **DELIBERATE.** |
| P3 | Lock-free row-partitioned fill; `DenseDistanceMatrix` has no locks and no atomics | `dtwc/Problem.cpp:805-807`; `dtwc/core/distance_matrix.hpp:38-41`; `.claude/design.md:25` | Any per-cell synchronisation serialises the whole N² kernel. **DELIBERATE.** |
| P4 | `wdtw_weights_cache_` filled **serially and exhaustively** before the parallel fill; never inserted into afterwards | `dtwc/Problem.cpp:271-274`; `dtwc/Problem.hpp:583-585` | Lazy/memoised insertion reinstates the fixed WDTW data race (`fdb69b3`). **DELIBERATE.** |
| P5 | Two-tier cache identity — one full data hash per session, fixed-size snapshot afterwards | `dtwc/Problem.cpp:618-623`; `docs/api-contract-2.0.md:546-548` | Re-hashing per lookup turns every O(1) warm read into O(series length). **DELIBERATE.** |
| P6 | `ensure_dense_cache_configuration_current_preflighted()` — a deliberate near-duplicate of its non-preflighted sibling | `dtwc/Problem.hpp:237-241`; `.claude/LESSONS.md:1119-1124` "Do NOT 'fix' it by dropping the outer call, which silently reorders the mmap-identity and semantics errors" | Merging the two either doubles the N² preflight cost or swaps which error a mis-bound cache reports. **DELIBERATE.** |
| P7 | `RelaxedFlag` — a hand-written movable wrapper over `atomic<bool>` | `dtwc/Problem.hpp:160-176` "`std::atomic` is neither copyable nor movable, so the value-moving members are what keep `Problem`'s `= default` move operations well-formed" | A plain `bool` is a data race under the released-GIL policy; a bare `std::atomic<bool>` makes `Problem`'s defaulted move operations ill-formed. **DELIBERATE.** |
| P8 | `dtw_binding_owner_` + `repair_dtw_binding_after_relocation()` | `dtwc/Problem.hpp:139`; `dtwc/Problem.cpp:447-460`; `.claude/LESSONS.md:738-743` "Default-moving a `std::function` does not rebind a lambda that captured `this` … A cached result can hide a live use-after-move path" | Deleting it as redundant bookkeeping reintroduces a use-after-move that returns **plausible cached values**, not a crash. **DELIBERATE.** |
| P9 | `series_storage_owner_` as `unique_ptr<LoadedData>` + pointer-identity assertions | `dtwc/Problem.hpp:268-294`; `.claude/LESSONS.md:744-750` "Equal values did not prove that spans and `string_view`s targeted the relocated mmap/name owner; short SSO names could even keep stale bytes looking valid" | Flattening to a by-value member, or relaxing the checks to value comparison, silently reinstates dangling views. **DELIBERATE.** |
| P10 | The dense matrix is **not** allocated by `set_data` / `refresh_distance_matrix` — allocation is deferred to the fill | `dtwc/Problem.cpp:230-240` "so that large-N algorithms (e.g. FastCLARA) can load data without forcing quadratic memory usage" | Eager allocation costs N²·8 bytes for matrix-free methods (`clara`, `onebatch`, `tadpole`). **DELIBERATE.** |
| P11 | Serial missing-data pre-scan before the parallel fill | `dtwc/Problem.cpp:888-918` "so the diagnostic can name the offending series … and no per-pair path has to throw" | Per-pair throwing inside an OpenMP region is UB (`.claude/LESSONS.md:99`). **DELIBERATE.** |
| P12 | `fillDistanceMatrix_BruteForce` resizes **only when the size differs** | `dtwc/Problem.cpp:788-796` "an unconditional resize discarded a restored checkpoint and recomputed every pair" | A NaN-fill on every entry throws away a restored checkpoint. **DELIBERATE.** |
| P13 | Guard ORDER in the fill: every downgrade **above** the f32 rejection | `dtwc/Problem.cpp:976-990`; `.claude/LESSONS.md:1162-1165` "A rejection must sit below every downgrade that would make it moot" | Reordering breaks Pruned+mmap+f32, which the downgrade already routes to an f32-capable exact fill. **DELIBERATE.** |
| P14 | Every fingerprint axis the caller controls is a **parameter**, never a literal | `dtwc/Problem.cpp:579-586`; `.claude/LESSONS.md:428-433` | Re-defaulting `MetricType` away lets an L1 run accept a SquaredL2 matrix — silent wrong answer. **DELIBERATE.** |
| P15 | `#pragma omp critical(distByInd_init)` is **named** | `dtwc/Problem.cpp:697`; `.claude/LESSONS.md:1062-1069` "An UNNAMED `#pragma omp critical` shares ONE implementation-defined name across the entire program" | Dropping the name globally serialises unrelated critical sections. **DELIBERATE.** |
| P16 | One relaxed `fetch_add` **per load call**, never per series (both DataLoader atomics) | `dtwc/DataLoader.hpp:139-153, 282, 519` | Per-series atomics would be a real cost in the loader. **DELIBERATE.** |
| P17 | `Problem::series(i)` hoisted once per row in the fill loop | `dtwc/Problem.cpp:809, 819` | Calling `series(i)` inside the inner loop pays the three-way branch N times per row. **DELIBERATE.** |
| P18 | `assign_clusters` drops to 1 worker when the matrix is not yet filled | `dtwc/Problem.cpp:1225-1231` | Parallelising the lazy-compute path races on the packed (i,j)/(j,i) slot. **DELIBERATE.** |
| P19 | `system_memory.cpp` exists purely so `<windows.h>` never reaches the umbrella header | `dtwc/system_memory.cpp:5-8`; `dtwc/DataLoader.hpp:39-42` | Inlining it leaks `ERROR`, `GetMessage`, `min` / `max` into every consumer TU. **DELIBERATE.** |

Contradiction for the designer: `.claude/design.md:24` says DTW is **latency-bound**; `MEMORY.md`
and `.claude/LESSONS.md:150-156` say **memory-bound**, with the caveat that "no PMU artifact proves
a memory-bound bottleneck". Neither is settled. `[not established]`

---

## 6. Const-correctness, mutable state, threading

**`mutable` members (7, all in `Problem`)** — `dtw_fn_` (`:137`), `dtw_fn_f32_` (`:138`),
`dtw_binding_owner_` (`:139`), `wdtw_weights_cache_` (`:140`), `mmap_cache_data_validated_`
(`:180`), `dense_cache_configuration_` (`:181`), `dense_cache_configuration_bound_` (`:182`).
All seven are *derived* state — recomputable from `data_` plus the configuration fields. No
`mutable` members exist anywhere else in scope `[confirmed: grep "mutable" over all 20 files]`.

**`const` methods that mutate.**

| Method | Mutates | Precondition it relies on | Hot/cold |
|---|---|---|---|
| `rebind_dtw_fn() const` `Problem.cpp:298-320` | all 7 mutables | Called serially | Hot when reached from `dist_by_ind`'s critical section (`:709`) |
| `refresh_variant_caches() const` `Problem.cpp:262-293` | `wdtw_weights_cache_` | Serial, before the parallel fill | Cold |
| `validate_mmap_cache_identity() const` `Problem.cpp:596-635` | `mmap_cache_data_validated_` | First use must be serial: `fill_distance_matrix()` or `is_distance_matrix_filled()` once before parallel lookups (`Problem.cpp:674-676`) | **Hot** — runs on every `dist_by_ind` |
| `validate_dtw_function_configuration() const` `Problem.cpp:508-526` | rebinds when only the address drifted (`:516-519`) | "Repairing that state is logically const" | Cold |
| `dtw_function() const`, `dtw_function_f32() const`, `max_distance() const`, `print_distance_matrix() const`, `distance_matrix() const`, `dense_distance_matrix() const`, `is_distance_matrix_filled() const`, `write_distance_matrix() const` | transitively, via the two validators above | as above | mixed |

**Double-checked pattern (one).** `dist_by_ind` `[confirmed dtwc/Problem.cpp:694-711]`:
`bool needs_init = visit_distmat(... m.size() != N)` is a **plain, unsynchronised read** that races
the write inside `#pragma omp critical(distByInd_init)`. Safety rests on the documented "prime one
non-diagonal distance serially first" precondition (`:671-676, 691-693`), not on an enforced
invariant. Still OPEN at HEAD; the io-cli review flagged the same lines
`[.claude/reports/2026-09-02-review-io-cli.md:45]`. Aggravating factor: `rebind_dtw_fn()` — which
writes all seven mutables — is called *inside* that critical section (`:709`), so any concurrent
reader of `dtw_fn_` races.

**OpenMP / atomics / statics in scope.**

- `#pragma omp critical(distByInd_init)` — `Problem.cpp:697`. Named. Hot-adjacent (guarded by the
  lock-free `needs_init` pre-check, so steady state never enters it).
- `#pragma omp parallel` — `test_api.hpp:152`. Each thread writes only its own slot; documented at
  `:147`. Cold (probe).
- `run_openmp(fill_row, N, true, 8)` — `Problem.cpp:828, 840`. Hot. Deterministically rethrows the
  lowest-row failure after the join (`:805-807`).
- `run(...)` — `Problem.cpp:1231, 1249, 1271`; `initialisation.cpp:112`; `scores.cpp:157`. Hot.
- `static std::atomic<std::size_t> counter` + `static const std::string process_tag` —
  `DataLoader.hpp:139-151`. One relaxed `fetch_add` per **load**. Cold.
- `static inline std::atomic<std::size_t> s_bulk_read_invocations` — `DataLoader.hpp:282`. Test
  instrumentation living in production code; one relaxed increment per load call (`:519`). Cold.
- `static std::atomic<std::uint64_t> sequence` — `checkpoint.cpp:378`. Once per generation. Cold.
- `std::once_flag g_seq_warned_flag` + `std::call_once` — `env.cpp:247, 264`. Constant-initialised
  (no static-init-order hazard, documented at `:244-246`). Cold.
- `Env &env()` — `env.cpp:370-373`. Meyers singleton; thread-safe initialisation.

**Const-correctness smell (Tier-1).** `Result::score()`, `Result::distance_matrix()` and
`Result::save()` are all `const` yet call the non-const `Problem::fill_distance_matrix()` and
`dist_by_ind()` through the `shared_ptr` member — constness does not propagate through a smart
pointer `[confirmed dtwc/api.cpp:216-249, 240, 246, 288]`. Legal, but the `const` promises nothing.

---

## 7. Duplication

| # | What | Where | Classification |
|---|---|---|---|
| U1 | **Three output-artefact schemas.** Tier-1 writes `<name>_labels.csv` (`name,cluster`), `<name>_medoids.csv` (`cluster,medoid_index,medoid_name`), `<name>_distance_matrix.csv`, `<name>_silhouettes.csv` (`name,cluster,silhouette`). The CLI writes the **same four filenames with the same headers** through its own private writers. `Problem_IO` writes a fourth, entirely different set: `<name>_Nc_<k>.csv`, `<name>_silhouettes_Nc_<k>.csv`, `<name>_distanceMatrix.csv`, `<name>medoids_rep_<r>.csv`, `<name>_bestRepetition_Nc_<k>.csv`, `medoidMembers_Nc_..._rep_..._iter_....csv` | api side: `dtwc/api.cpp:263-319`; CLI side: `dtwc/dtwc_cl.cpp:634-698, 1854-1880` (cited from `.claude/reports/2026-09-02-review-io-cli.md:54`, only the headers spot-checked here); Problem side: `dtwc/Problem_IO.cpp:72-95, 124-147, 158-180, 187-203, 242-251` | **near-identical** between api.cpp and dtwc_cl.cpp (same names, same headers, different code); **semantic** divergence for `Problem_IO` (different filenames, different content — a `"Procedure is completed with cost:"` trailer that neither other writer emits) |
| U2 | Silhouette-skip warning text and policy, written out twice verbatim: `"Warning: silhouettes skipped: "` | `dtwc/Problem_IO.cpp:161-166` and `dtwc/api.cpp:305-311` | **byte-identical** (message + `catch (const UndefinedScore&)` + `return`) |
| U3 | The mmap CSV emitter is open-coded in `Problem_IO` because `MmapDistanceMatrix` has no `write_csv` | `dtwc/Problem_IO.cpp:219-234` vs `io::write_csv` used at `Problem_IO.cpp:218` | **near** — same bytes by construction (the comment at `:222-224` states the intent), duplicated only to hoist the O(N²) preflight above the truncating open |
| U4 | Distance-matrix CSV loop still exists twice in `core/matrix_io.hpp` | `core/matrix_io.hpp:106-115` and `:213-220` | **near** (the third copy, in `Problem_IO`, was removed by `e259de9`) |
| U5 | `Env::threads()` is byte-identical to the anonymous-namespace `effective_max_threads()` | `dtwc/env.cpp:251-258` vs `:361-368` | **byte-identical** bodies |
| U6 | The `preflight_*` pair: `preflight_current_distance_semantics` and `preflight_float32_distance_semantics` differ only in a trailing `true` | `dtwc/Problem.cpp:424-434` | **near** — trivial, and deliberate |
| U7 | Four `Data` constructors repeat the same size-mismatch check and `validate_ndim()` call | `dtwc/Data.hpp:116-160` | **near** |
| U8 | `random`/`random_seeded` and `Kmeanspp`/`Kmeanspp_seeded` are already deduplicated behind `random_with`/`kmeanspp_with`; only the RNG lambdas differ | `dtwc/initialisation.cpp:57-120` vs `137-214` | **not duplication** — noted so it is not "fixed" |
| U9 | `cluster_counts_checked` + `require_two_realised` are correctly shared by silhouette/DBI/dunn/CH | `dtwc/scores.cpp:45-86` | **not duplication** — the good pattern in this file |
| U10 | Validation repeated at three layers for the same values: Tier-1 (`validate_common`, `normalize_method`), `Problem` setters (`preflight_*`, `validate_*`), and `core::validate_*` | `dtwc/api.cpp:62-92`; `dtwc/Problem.hpp:57-112, 438-536`; `dtwc/core/distance_semantics.hpp:91` | **semantic** — layered by design for Tier-1/Tier-2 parity, but see D1 for the cost |

---

## 8. Design problems

| # | Problem | Evidence | Severity | Blast radius |
|---|---|---|---|---|
| **D1** | **The comment "Exactly ONE preflight per call" is false: `dist_by_ind` preflights TWICE.** It calls `preflight_current_distance_semantics()` directly (`:682`) and then `validate_mmap_cache_identity()`, whose first statement is another `preflight_current_distance_semantics()` (`:598`). Each preflight runs `validate_precision` + `validate_distance_matrix_strategy` + `validate_cuda_settings_precision` + `validate_variant_params` (8 sub-validators) + `validate_variant_missing_semantics` + six more branches. The `..._preflighted()` sibling removed the *other* duplicate; this one survived. | `dtwc/Problem.cpp:678-684` vs `:596-599`; `dtwc/core/distance_semantics.hpp:91-124`; `.claude/LESSONS.md:1119-1124` | **High** | Every N² kernel: Lloyd assignment, `find_total_cost`, `distanceInClusters`, `calculate_medoids`, all `scores::*`, `Result::distance_matrix()`. Candidate action: give `validate_mmap_cache_identity` a `..._preflighted` sibling exactly as the dense path has, preserving the error order. |
| **D2** | **God object.** ~174 declared members across 8 responsibility clusters; `Problem.hpp` is 704 lines carrying three settings structs, an enum, a nested `RelaxedFlag` class, two nested config structs and five free validators. Algorithms in `dtwc/algorithms/*` reach in and assign `prob.clusters_ind` / `prob.centroids_ind` directly — an undocumented cross-module writeback invariant that Tier-1 `Result` silently depends on. | `dtwc/Problem.hpp:122-701`; `dtwc/algorithms/fast_pam.cpp:537-539`, `one_batch_pam.cpp:396-398`, `fast_clara.cpp:568-570`, `hierarchical.cpp:279-281`; `dtwc/api.cpp:213-214` | **High** | Everything. Candidate action: extract C2+C3 into an owned `DistanceCache` component behind the unchanged public setters (no private member is in the frozen surface, §10). |
| **D3** | **`set_data` does not resize the label buffers; `set_view_data` does.** After `set_n_clusters(k); set_data(bigger)`, `clusters_ind.size() != size()`. The inline comment on the `set_view_data` call is also wrong: `resize()` sizes `clusters_ind` / `centroids_ind`, not the distance matrix. | `dtwc/Problem.hpp:509-536` (no `resize()` at `:522`; `resize(); // sizes distance matrix for new N` at `:535`) vs `dtwc/Problem.cpp:120-124` | **Medium** | Caught downstream by `scores::cluster_counts_checked` (`scores.cpp:49-52`), so it surfaces as a confusing `InvalidInput` rather than a wrong answer. Candidate action: call `resize()` from `set_data` too and fix the comment. |
| **D4** | **`distanceInClusters()` is pure wasted work in the Lloyd loop.** `cluster_by_kmedoids_lloyd_impl` calls `fill_distance_matrix()` unconditionally before the repetition loop, so every pair is already computed; `distanceInClusters()` then issues ≈ N²/(2k) `dist_by_ind` calls per iteration that all hit the "already computed" branch — each paying the D1 double preflight. | `dtwc/Problem.cpp:1321` (`fill_distance_matrix();`), `:1408` (`distanceInClusters(); // Just populates distance matrix ahead.`), `:1240-1250`; sole caller confirmed by grep | **Medium** | Lloyd only. Candidate action: delete the call (and the private method — it has no other caller). |
| **D5** | **String-typed dispatch across the Tier-1 boundary.** `cluster()` takes `std::string_view method` and `std::string_view device`, validates the method against a string array, then re-parses the same string into `Method` enum values 70 lines later. `MIPSettings::benders` is a `std::string` that reaches the library and is compared to literals at dispatch time. | `dtwc/api.cpp:76-92, 326, 366-402`; `dtwc/Problem.hpp:72`; `dtwc/Problem.cpp:1178` | **Medium** | Tier-1 + CLI + both bindings. Candidate action: parse once into an enum at the boundary and dispatch on the enum. |
| **D6** | **`writeMedoids` is the one writer that bypasses the shared open/close helpers** — it opens raw, checks `good()`, prints to `std::cout` on failure, and calls `close()` **unchecked**, so a full disk silently loses the file. | `dtwc/Problem_IO.cpp:76-94` vs the helpers at `:47-62` used everywhere else (`:128, 173, 193, 246`) | **Medium** | `cluster_and_process` artefacts only. Candidate action: route it through `open_output` / `close_output`. |
| **D7** | **`set_max_iter` / `set_n_repetitions` perform no validation** and write directly into the deprecated public fields. `set_max_iter(0)` yields a Lloyd run whose iteration loop never executes: `status` stays −1, `assign_clusters()` runs once, and a cost is reported as if converged. Tier-1 validates `max_iter > 0`, so only Tier-2 and the bindings are exposed. | `dtwc/Problem.cpp:153-171`, `:1388-1423`; contrast `:1314-1315`, which *does* reject `n_repetitions <= 0`; `dtwc/api.cpp:73` | **Medium** | Tier-2 C++, Python (`set_max_iter`), MATLAB. Candidate action: reject `n < 1` in both setters. |
| **D8** | **Wide constructor interface: `Problem(std::string_view, DataLoader &)` takes a non-const lvalue reference** purely because `DataLoader`'s getters are non-`const`. A temporary loader will not bind. | `dtwc/Problem.hpp:344`; `dtwc/DataLoader.hpp:295-304` (only `storage_policy()` is `const`) | **Low** | `main.cpp:20-23` and any Tier-2 caller. Candidate action: make the DataLoader getters `const`. |
| **D9** | **Three friends puncture the encapsulation**, one of them for tests only. `load_checkpoint` reaches `prob.distMat` directly; `MIP_clustering_byBenders` needs `cluster_by_kmedoids_lloyd_impl(false)`; `ProblemStoragePolicyTestAccess` exists solely for `tests/unit/unit_test_problem_storage_policy.cpp:47`. | `dtwc/Problem.hpp:253-256`; `dtwc/checkpoint.cpp:601`; `tests/unit/unit_test_problem_storage_policy.cpp:47, 88` | **Low** | Any decomposition of `Problem` must keep all three working. |
| **D10** | **Settings sprawl.** `Problem` carries `CUDASettings`, `MIPSettings`, `CheckpointOptions` plus 9 loose configuration fields plus `storage_policy_` / `ram_limit_bytes_` that duplicate `DataLoader`'s. There is still no shared config representation outside the CLI, so Python and MATLAB cannot load a config file at all. | `dtwc/Problem.hpp:48-74, 189-190, 307-319`; `dtwc/DataLoader.hpp:275-277`; `.claude/design.md:62-64`; `.claude/reports/2026-09-02-review-io-cli.md:76` (F2, still OPEN) | **Medium** | All front-ends. |
| **D11** | **Contract drift: `Problem` now has 12 public data members, but the frozen contract enumerates 11.** `CheckpointOptions checkpoint` (added by `a5b9f5f`) is not in the contract's list. | `dtwc/Problem.hpp:304-322` (12) vs `docs/api-contract-2.0.md:264-269` (enumerates exactly 11) | **Low** | Governance: `docs/api-contract-2.0.md:29` requires an explicit dated decision for non-additive edits. Candidate action: record the addendum. |
| **D12** | **Hidden coupling: `DataLoader::load()` reads the process-wide `dtwc::env()`.** A pure-looking loader silently changes behaviour with a global. `load_local()` exists precisely to escape it — an admission of the coupling. | `dtwc/DataLoader.hpp:420-425, 427-435, 467-471` | **Medium** | Every loader caller; already worked around in Tier-1 (`api.cpp:158`). |
| **D13** | **`route_series_storage` pays an O(N) footprint scan plus a free-RAM syscall even when the policy is `Heap`.** Both results are consumed only on the mmap branch and in the f32 warning. | `dtwc/DataLoader.hpp:181-194` | **Low** | Every `set_data` / `load_stored` with an explicit `Heap` policy. Candidate action: compute lazily inside the `want_mmap` branch. |
| **D14** | **Tier-1 in-memory clustering copies the entire dataset.** `materialize_local()` is `const` and `cluster()` takes `const Dataset&`, so `std::get<series_type>(source_)` copies every series before the slicing edits. This is the one place in the orchestration layer that violates the "data is never copied" principle. | `dtwc/api.cpp:151-183` (`:166`), `:322` | **Medium** | The MATLAB binding's in-memory route (`dtwc::load(series_type, …)`, `dtwc_mex.cpp:1484`) pays it on every call. Candidate action: add an rvalue/consuming overload of `cluster`. |
| **D15** | **`Result::distance_matrix()` materialises N² through `dist_by_ind`**, paying the full guard chain (including D1) per element, after having already filled the matrix. | `dtwc/api.cpp:236-249` | **Low-Medium** | Python `Result.distance_matrix` parity path. Candidate action: read the packed matrix directly. |
| **D16** | **`is_distance_matrix_filled()` is O(N²)** (`all_computed()` scans every packed slot for NaN) and is called once per Lloyd iteration inside `assign_clusters` to choose the worker count. | `dtwc/Problem.cpp:590-597, 1230`; `dtwc/core/distance_matrix.hpp:93-96` | **Low** | Dominated by the assignment loop itself, but it is an avoidable O(N²) scan per iteration. |

---

## 9. Obsolete / dead code, stale comments, docs

### 9.1 The `[[deprecated]]` shims — exact count and caller proof

Repo-wide there are **33** real `[[deprecated]]` attributes (a raw grep returns 35 lines; two are
prose: `dtwc/core/llfio_include.hpp:13` and `dtwc/scores.hpp:24`). Distribution:
`Problem.hpp` **22**, `scores.hpp` **5**, `DataLoader.hpp` **2**, `settings.hpp` **4**.
The contract's own figure agrees: "33 C++ diagnostic entities"
`[confirmed docs/api-contract-2.0.md:669]`. The brief's figure of 34 is one high.

`Problem.hpp` sites: `:303, 305, 366, 408, 414, 418, 548, 551, 598, 645, 648, 656, 661, 665, 668,
672, 678, 687, 689, 695, 697, 700`. `scores.hpp`: `:38, 41, 44, 47, 54`.
`DataLoader.hpp`: `:330, 339`.

**Proof of zero non-test callers.** Command run for all 27 deprecated names:

```
grep -rn "\b<name>\b" --include=*.cpp --include=*.hpp --include=*.cu --include=*.cuh \
     dtwc/ benchmarks/ python/ bindings/ examples/ | grep -v deprecated
```

Result: every match is either the declaration itself or a forwarding body. The only two substantive
hits are:

- `maxIter` — 3 hits: the declaration (`Problem.hpp:304`) and **its own canonical accessors**
  (`Problem.cpp:155, 160`). Same for `N_repetition` (`Problem.hpp:306`; `Problem.cpp:165, 170`).
  These are not callers: the deprecated public field *is the storage* for the canonical accessor,
  which is why both sites sit inside a `-Wdeprecated-declarations` push/pop
  `[confirmed dtwc/Problem.cpp:138-179]` and why the class's in-class initialisers need the same
  push/pop around every constructor `[confirmed dtwc/Problem.hpp:325-357]`.
- `cluster_size` — the Python binding exposes the *name* but implements it with a lambda calling
  `n_clusters()`, so the C++ shim is not called `[confirmed python/src/_dtwcpp_core.cpp:980]`.

Everything else (`refreshDistanceMatrix`, `set_numberOfClusters`, `maxDistance`, `distByInd`,
`isDistanceMatrixFilled`, `fillDistanceMatrix`, `printDistanceMatrix`, `cluster_by_MIP`,
`cluster_by_kMedoidsLloyd`, `findTotalCost`, `assignClusters`, `calculateMedoids`,
`readDistanceMatrix`, `writeDistanceMatrix` ×2, `printClusters`, `writeClusters`,
`writeMedoidMembers`, `writeSilhouettes`, `startColumn`, `startRow`, and the five `scores::*`
aliases) has **zero** callers outside `tests/` and `scripts/`. They are retained on purpose:
`docs/api-contract-2.0.md:640-645` mandates them until 3.0, and
`scripts/test_f22_cpp_deprecations.py` asserts each one still warns. **Keep.**
`design.md:51-52` argues the opposite for binding-level helpers, not for these C++ shims.

### 9.2 Dead / unreachable code

- `Problem::distanceInClusters()` — one caller, and that call is redundant (D4)
  `[confirmed dtwc/Problem.cpp:1240-1250, 1408]`.
- `LowerBoundStrategy::None` inside the Pruned path is unreachable: `Problem.cpp:1026-1031`
  short-circuits it to BruteForce before `core::fill_distance_matrix_pruned` is entered, yet
  `core/pruned_distance_matrix.cpp:87-95` handles it. **OPEN** (carried from
  `.claude/reports/2026-09-02-review-core.md:235-236`).
- `checkpoint.cpp:254-255` — `n > std::numeric_limits<std::size_t>::max()` on a `std::uint64_t` is
  always false on a 64-bit host. **OPEN.**
- `checkpoint.cpp:929-935` — both `catch` clauses (`fs::filesystem_error`, `ios_base::failure`) are
  unreachable: the code uses `error_code` overloads and never enables stream exceptions. **OPEN.**
- `api.cpp:117-118` — two defensive `logic_error` throws for states `configure_device` cannot reach.
  Harmless; flagged for completeness. **OPEN.**
- `DataLoader::s_bulk_read_invocations`, `bulk_read_count()`, `reset_bulk_read_count()` —
  test-only instrumentation compiled into the production library
  `[confirmed dtwc/DataLoader.hpp:279-315]`. **CHANGED** since the io-cli review (now atomic and
  once-per-load) but not removed.
- `ProblemStoragePolicyTestAccess` — a `friend` declaration in production for a type defined only in
  `tests/unit/unit_test_problem_storage_policy.cpp:47`. **OPEN.**

### 9.3 Stale / wrong comments

- `Problem.cpp:680` — *"Exactly ONE preflight per call"*. False at HEAD; there are two (D1).
- `Problem.hpp:535` — `resize(); // sizes distance matrix for new N`. `resize()` does not touch the
  distance matrix (D3).
- `env.cpp:86` — *"requires a .env file at the repository root"*. The search directory is
  `$DTWC_REPO_ROOT` **or the current working directory** `[confirmed dtwc/env.cpp:228-234]`; the
  `error_code` from `fs::current_path` is captured and never inspected. **OPEN**
  (`.claude/reports/2026-09-02-review-io-cli.md:37`).
- `main.cpp:19-20` — *"Run this from the project root directory"* with `fs::path("data") / "dummy"`.
  Directly violates non-negotiable #1 (no runtime dependence on repo-relative paths). **OPEN.**
- `Problem.hpp:207-208` / `Problem_IO.cpp:210` — `write_distance_matrix(const std::string &name_)`
  still shadows the member `name_`; now documented rather than fixed. **OPEN (documented).**
- `checkpoint.hpp:83, 101` — `save_checkpoint` / `load_checkpoint` take `const std::string &path`
  while the binary API takes `fs::path`; no `utf8_to_path` anywhere in `checkpoint.cpp`, so
  non-ASCII Windows paths are mangled. **OPEN.**
- `checkpoint.cpp:58` bounds the reader at `MAX_NUMERIC_TOKEN_SIZE = 32` while the writer uses an
  unchecked `std::array<char, 64>` (`:529`). Asymmetric bound. **OPEN.**
- `scores.cpp:3` — the file header says "Header file for calculating…" in a `.cpp`. Cosmetic.
- `initialisation.cpp:1` — `#include <stdexcept>` sits **above** the file's doc comment. Cosmetic.

### 9.4 `test_api.hpp` — naming vs role

The name says "test API". The file is **production surface**: it is included by
`python/src/_dtwcpp_core.cpp:51` and `bindings/matlab/dtwc_mex.cpp:38` and exposed as
`dtwcpp.test.parallelisation()` / `dtwcpp.test.gpu()` and
`dtwc_mex('test_parallelisation'|'test_gpu')`
`[confirmed python/src/_dtwcpp_core.cpp:1788-1829; bindings/matlab/dtwc_mex.cpp:834]`. It is a
*self-introspection / capability-probe* API, not test scaffolding. It is also the only header in
scope that includes the whole `dtwc.hpp` umbrella (`test_api.hpp:35`), so every binding TU pulls in
all 38 umbrella headers to get two probe functions. Candidate action: rename to
`introspection.hpp` (or `capabilities.hpp`) and depend on the two headers it actually needs.

### 9.5 Prior-review items in scope — status at HEAD

**FIXED:** read-matrix exception swallowing (`Problem_IO.cpp:261-272`, `e259de9`);
metric-parameterised checkpoint identity (`Problem.cpp:579-586`, `e259de9`); mid-fill checkpointing
now real (`Problem.cpp:827-844`, `a5b9f5f`); generation pruning (`checkpoint.cpp:578-591`,
`e259de9`); `CheckpointOptions` now consumed (`Problem.hpp:317`, `a5b9f5f`); `ndim == 0` rejected
(`Data.hpp:100-101`, `397db9f`); precision-mismatch accessors (`Data.hpp:56, 65`); `path()` no
longer overrides an explicit delimiter (`DataLoader.hpp:369-377`); racy temp-path counter
(`DataLoader.hpp:139-153`); loud f32+Mmap rejection (`DataLoader.hpp:197-207`, `2ef94c9`);
`Problem_IO` stream checking (`Problem_IO.cpp:39-62`, `e259de9`); `skip_rows` on Tier-1 `load`
(`api.hpp:58-59`, `397db9f`); `--benders` validation (`Problem.hpp:88-89`, `397db9f`); all-NaN
Interpolate pre-scan (`Problem.cpp:903-917`, `e1cb8a3`); pruned fill no longer wipes a restored
checkpoint (`core/pruned_distance_matrix.cpp:80-85`, `e1cb8a3`); `verbosity(0)` now silences the
loader (`fileOperations.hpp:388-475`, `397db9f`); `MissingStrategy::Error` enforced outside
`Problem` (`distance.hpp:117-118`, `e1cb8a3`).

**OPEN:** `dist_by_ind` double-checked lock without atomic/fence (`Problem.cpp:694-707`);
`rebind_dtw_fn() const` mutating shared state inside that critical section (`Problem.cpp:298-320`,
invoked at `:709`); binary checkpoint save non-atomic (`checkpoint.cpp:900-901`); `Env::threads()`
duplicated (`env.cpp:251-258` vs `:361-368`); three output schemas (U1); string-typed dispatch (D5);
`LowerBoundStrategy::None` unreachable; `name_` shadowing; `checkpoint` path types; `.env` message;
`main.cpp` relative path; WDTW hash lookup + heap alloc per pair (`core/dtw_dispatch.cpp:173, 193`);
`compute_distance_matrix_pruned` takes no `LowerBoundStrategy`
(`core/pruned_distance_matrix.hpp:89-94`); no NaN policy at load; `std::function` per-pair indirect
call (open **by design**, see P1). `[all confirmed at the cited lines]`

**CHANGED (not fixed):** stale WDTW weights are now caught by the dense-configuration fingerprint
rather than by encapsulating `variant_params` (`Problem.cpp:356, 468-479`) — the code path was read,
**not** exercised; a test that writes `prob.variant_params.wdtw_g` directly and checks the resulting
weights would settle it `[not established]`. `s_bulk_read_invocations` is now atomic but still ships
in production. The CSV emission count fell from four copies to three; api.cpp and dtwc_cl.cpp now
agree on filenames and headers.

---

## 10. Frozen surface (do not change this campaign)

Grep-confirmed against `python/src/_dtwcpp_core.cpp` (1833 lines) and
`bindings/matlab/dtwc_mex.cpp` (1797 lines), both read in full. **PY** = python line,
**MEX** = matlab line.

**`Problem` — 63 distinct members (61 excluding the two constructor forms).**

*Constructors:* `Problem()` PY 897; `Problem(std::string)` PY 899, MEX 542/1216/1575.

*Public data read/written directly:* `band` PY 924, MEX 568/626/1217/1576 — *note MEX 1217 and 1576
write the raw field, bypassing `set_band`*; `variant_params` PY 928, MEX 674; `missing_strategy`
PY 934; `distance_strategy` PY 940; `cuda_settings` PY 955, MEX 1002/1011; `mip_settings` PY 961,
MEX 921/935/942; `checkpoint` PY 963, MEX 973/983/989; `clusters_ind` PY 975, MEX 725;
`centroids_ind` PY 976, MEX 719.

*Setters:* `set_band` PY 925/1018, MEX 620; `set_method` PY 902/1017, MEX 877; `set_max_iter`
PY 904/1019, MEX 640/1577; `set_n_repetitions` PY 906/917/1020, MEX 646; `set_random_seed`
PY 921/1021; `set_variant` (both overloads) PY 931/1022/1024, MEX 695; `set_missing_strategy`
PY 936, MEX 659; `set_distance_strategy` PY 942, MEX 666; `set_lb_strategy` PY 946, MEX 893;
`set_storage_policy` PY 950, MEX 901; `set_cuda_settings` PY 958, MEX 1005; `set_verbose` PY 969,
MEX 543/634/1218/1578; `set_name` PY 971; `set_output_folder` PY 973, MEX 908; `set_n_clusters`
PY 1010/1014, MEX 652; `set_solver` PY 1026, MEX 884; `set_data` PY 1031/1034, MEX 614/1220/1581;
`set_view_data` PY 1039.

*Read accessors:* `method()` PY 902; `max_iter()` PY 903, MEX 570; `n_repetitions()` PY 905/912,
MEX 572; `random_seed()` PY 920; `lb_strategy()` PY 945; `storage_policy()` PY 949; `verbose()`
PY 968, MEX 569; `name()` PY 971/1117, MEX 566/713; `output_folder()` PY 972; `size()`
PY 880/978/1117, MEX 567/701/791; `n_clusters()` PY 979/983/1118, MEX 707/1041; `labels()` PY 986;
`medoids()` PY 988; `series(size_t)` PY 991; `series_name(size_t)` PY 995; `centroid_of(int)`
PY 997.

*Matrix + compute:* `is_distance_matrix_filled()` PY 999, MEX 573/731; `max_distance()` PY 1000,
MEX 1035; `dist_by_ind(int,int)` PY 1003, MEX 750/1228; `fill_distance_matrix()` PY 863/1047,
MEX 741/1221; `dense_distance_matrix()` (const + non-const) PY 864/882, MEX 771/794;
`refresh_distance_matrix()` PY 1069, MEX 1022; `read_distance_matrix(path)` PY 1072, MEX 1029;
`print_distance_matrix()` PY 1077; `use_mmap_distance_matrix(path)` PY 1081.

*Clustering + IO:* `cluster()` PY 1087, MEX 757; `find_total_cost()` PY 1091, MEX 763;
`assign_clusters()` PY 1095; `calculate_medoids()` PY 1099; `print_clusters()` PY 1102;
`write_clusters()` PY 1105; `write_medoid_members(int,int)` PY 1107; `write_distance_matrix()`
PY 1110; `write_silhouettes()` PY 1114.

**`Data` — 11 members.** `Data()` PY 763/835; f64 ctor PY 764/767/1030/1038, MEX 613/1219/1580;
f32 ctor PY 774; `p_vec` PY 779; `p_names` PY 780; `ndim` PY 781; `size()` PY 783; `is_f32()`
PY 784; `is_view()` PY 785; `series_length(size_t)` PY 786; `validate_ndim()` PY 788.

**`DataLoader` — 6 members, Python only** (`grep -c "dtwc::DataLoader" dtwc_mex.cpp` = 0):
`DataLoader(const fs::path&)` PY 242; `start_column(int)` PY 243; `start_row(int)` PY 243;
`verbosity(int)` PY 243; `delimiter(char)` PY 244; `load_local()` PY 246.

**Tier-1 `api.hpp` — 14 entities.** `device(string_view)` PY 220; `device()` PY 225;
`load(path, …)` MEX 1482; `load(series_type, …)` MEX 1484; `cluster(...)` MEX 1488; `Dataset`
MEX 1481; `Dataset::name()` MEX 1502; `Result` MEX 1488/1490/1508/1516/1526/1598;
`Result::labels()` MEX 1498; `Result::medoids()` MEX 1499; `Result::cost()` MEX 1500;
`Result::device()` MEX 1501; `Result::score(string_view)` MEX 1517; `Result::save(path)` MEX 1527.

**Total frozen surface: 94 entities** (63 + 11 + 6 + 14).

Additional constraints the designer must respect:

1. **`core::DenseDistanceMatrix` is transitively frozen.** Both bindings call `.size()`, `.get(i,j)`,
   `.set(i,j,v)`, `.resize(n)` through `Problem::dense_distance_matrix()` (PY 865/883/887,
   MEX 772/778/795/799); Python also binds the class directly (PY 608-650).
2. **The Tier-1 split is asymmetric.** MATLAB is the *only* consumer of
   `dtwc::load` / `cluster` / `Dataset` / `Result`; Python never touches them (its route is
   `DataLoader` → `Problem`). Freezing either half protects exactly one language.
3. **Not used by either binding, therefore free to change:** `Dataset::is_path()`, `::path()`,
   `::skip_cols()`, `::skip_rows()`, `::delimiter()`, `Result::distance_matrix()`.
4. **`ClusterOptions` does not exist** — `dtwc::cluster` takes six loose scalar/string parameters
   (`api.hpp:109-110`). Nothing to freeze under that name.
5. **No private `Problem` member is in the frozen set.** Every entity above is public, so a
   decomposition that keeps the public signatures identical is unconstrained by this list.

---

## 11. Open questions for the designer

1. **Is `Problem` decomposable behind its unchanged public setters?** §2 shows the private state
   partitions cleanly into C2 (binding) + C3 (cache/identity) + C1 (ownership); §10 confirms no
   private member is frozen. The obstacles are the three `friend` declarations (D9) and the fact
   that `dtw_fn_`'s closures capture `*this` and read C7's public fields at call time (P1/P8) — an
   owned `DistanceCache` component would need either a back-reference or a snapshot, and a snapshot
   changes the "reads live configuration at call time" semantics. Which of those two?
2. **Should the D1 double preflight be removed by a `validate_mmap_cache_identity_preflighted()`
   sibling, or by hoisting the whole guard chain out of `dist_by_ind` into a scoped "session" object
   the caller opens once before an N² loop?** The second is the bigger win but changes the public
   contract.
3. **Is per-call semantic revalidation the right model at all?** Today every O(1) matrix lookup pays
   ~30-40 branches of configuration validation so that raw public-field writes (`prob.band = -1`,
   `main.cpp:30`, `dtwc_mex.cpp:1217/1576`) are detected. Making the nine remaining public
   configuration fields private would remove the need — but §10 freezes exactly those fields this
   campaign.
4. **`Data` has four modes** (heap-f64, heap-f32, view, metadata-only) selected by three independent
   booleans/enums, giving 4 valid states out of 8 representable ones. Is a `std::variant` of four
   storage types viable given that `size()`, `series()`, `series_flat_size()` and `name()` are on
   hot paths and P2 requires the branch to remain a predictable branch?
5. **Who owns the output schema?** Three writers exist (U1); Tier-1 and the CLI now agree on
   filenames and headers while `Problem_IO` does not. Is `Problem_IO`'s schema a compatibility
   surface, or can `Problem::write_*` be re-pointed at the Tier-1 schema? `[not established]`
6. **Should `DataLoader::load()`'s dependence on the global `Env` be inverted** (device passed in)
   rather than papered over by `load_local()` (D12)?
7. **Is `latency-bound` or `memory-bound` the operative model for DTW?** `design.md:24` and
   `LESSONS.md:150-156` disagree, and no PMU artifact settles it. This determines whether the
   D1/D16 guard-chain costs matter at all relative to the kernel. `[not established]`
8. **Is `distanceInClusters` (D4) genuinely dead, or does some caller rely on it warming an mmap
   page cache?** Grep says one caller; a Lloyd run over a large mmap-backed matrix would settle it.
9. **`checkpoint` is a 12th public field not in the frozen contract's list of eleven (D11).** Does
   the campaign record an addendum, or is that field re-privatised?
10. **Several structures have their rationale recorded ONLY in header comments** —
    `persist_run_artifacts_`, `series_storage_owner_`, `dense_cache_configuration_`,
    `ensure_dense_cache_configuration_current_preflighted`, `dtw_binding_owner_`, `RelaxedFlag`,
    `is_metadata_only_` appear by name in neither `CHANGELOG.md` nor `.claude/LESSONS.md`, and the
    commits that introduced `dtw_binding_owner_` (`f1ef5e0`) and `series_storage_owner_`
    (`aa9781c`) have empty bodies. A comment-stripping "cleanup" would erase the sole record.
    Should §5 of this report be promoted into `design.md`?
