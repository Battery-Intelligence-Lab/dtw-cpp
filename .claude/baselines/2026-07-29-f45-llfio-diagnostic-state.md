# F45 — contain LLFIO/quickcpplib diagnostic state

Date: 2026-07-29 (Europe/London)

## Subject and inherited state

Base commit:

```text
b70a259d46486264f2a22d07cfd1b0dfcebdf953
```

F22's permanent public-header fixture exposed a prerequisite outside the alias
implementations. In the canonical LLFIO-ON compile context, including
`<dtwc.hpp>` makes Clang ignore every subsequent
`-Wdeprecated-declarations` diagnostic. The unmasked F22 driver printed:

```text
F22_CPP_SILENT count=33/33 entities=Problem::set_numberOfClusters(int),Problem::refreshDistanceMatrix(),Problem::readDistanceMatrix(const fs::path&),Problem::maxDistance() const,Problem::distByInd(int,int),Problem::isDistanceMatrixFilled() const,Problem::fillDistanceMatrix(),Problem::printDistanceMatrix() const,Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::findTotalCost(),Problem::assignClusters(),Problem::calculateMedoids(),Problem::cluster_by_MIP(),Problem::cluster_by_kMedoidsLloyd(),Problem::cluster_size() const,scores::daviesBouldinIndex(Problem&),scores::dunnIndex(Problem&),scores::calinskiHarabaszIndex(Problem&),scores::adjustedRandIndex(labels,labels),scores::normalizedMutualInformation(labels,labels),DataLoader::startColumn(int),DataLoader::startRow(int),settings::paths::setDataPath(const fs::path&),settings::paths::setDataPath(const char*),settings::paths::setResultsPath(const fs::path&),settings::paths::setResultsPath(const char*),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=0/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=FAIL
```

The command exited 1 by design. Its only compiler output was LLFIO's unrelated
`ntkernel-error-category/config.hpp:49` pragma message; there were zero
deprecation lines.

The otherwise-identical mandatory core/LLFIO-OFF arbiter printed:

```text
F22_CPP_SILENT count=9/33 entities=Problem::readDistanceMatrix(const fs::path&),Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=24/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=EXPECTED_RED
```

This disagreement is not an alias result. It proves the optional dependency
changes the caller's compiler diagnostic state.

## Localization

The exact installed dependency file is:

```text
build/highs-1151/install/include/quickcpplib/ringbuffer_log.hpp
SHA-256 51E6DC345219215D3F5818E77AF7BEAF506AA3996E82F20D5AB79622DF84F14D
```

Its lines 41–45 are:

```cpp
// If I'm on winclang, I can't stop the deprecation warnings from MSVCRT unless I do this
#if defined(_MSC_VER) && defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
```

There is no matching diagnostic push/pop. LLFIO includes that header at
`llfio/v2.0/config.hpp:210`. Independent probes under the configured Clang
21.1.8 context produced:

```text
no include:                         exit 1, custom deprecated use diagnosed
include llfio/v2.0/config.hpp:       exit 0, custom deprecated use silent
include llfio/v2.0/llfio.hpp:        exit 0, custom deprecated use silent
include ntkernel_category.hpp only:  exit 1, custom deprecated use diagnosed
```

Wrapping the LLFIO include in a Clang diagnostic push/pop restores the custom
diagnostic. The upstream header is generated dependency state and is not
edited.

Verdict: **F45 INHERITED RED [confirmed]**. This is a public-header containment
bug: canonical LLFIO-ON consumers cannot observe any DTWC++ deprecation
attribute declared after the optional include.

## Registered repair

These bands are written before any product edit.

1. Add one internal wrapper header for `<llfio/v2.0/llfio.hpp>`. Under Clang it
   pushes diagnostic state immediately before the third-party include and pops
   immediately after it. Other compilers receive the unmodified include.
2. Both and only the two current raw include sites,
   `core/mmap_data_store.hpp` and `core/mmap_distance_matrix.hpp`, include the
   wrapper. The wrapper contains the sole raw LLFIO include in tracked product
   headers.
3. A permanent compiler gate table-drives:
   - the wrapper directly;
   - `mmap_data_store.hpp`;
   - `mmap_distance_matrix.hpp`;
   - the LLFIO-OFF control.
   Each LLFIO-ON header is followed by a synthetic deprecated declaration/use
   and must fail only with that registered diagnostic under
   `-Werror=deprecated-declarations`. The OFF control compiles without optional
   headers and retains its own diagnostic.
4. The canonical LLFIO-ON F22 driver must move from 0/33 to the same exact
   24/33 expected-red ledger as llfio-OFF. Canonical names remain at zero
   deprecation lines. This is the decisive no-mask result before F22 alias
   product work.
5. Three mutations must be killed: remove the wrapper pop; restore the raw
   include in the data-store header; restore the raw include in the
   distance-matrix header. Exact bytes and hashes are restored after each.
6. Build and run the full canonical and llfio-OFF CTest matrices with their
   pre-F22 counts (122/122 and exact six/nine capability skips). This header
   containment change has no runtime behavior claim.
7. `CHANGELOG.md` records that public LLFIO-ON headers no longer suppress
   downstream Clang deprecation diagnostics. Add no CMake manifest or test
   target.

F45 has at most two product attempts. It is a new prerequisite finding with
its own registered band, not an expansion or reset of F22's two-attempt alias
repair cap. F22 resumes only after the canonical and llfio-OFF 24/33 ledgers
are identical.

Rollback is the eventual local F45 product commit. No dependency cache,
upstream source, remote, or irreversible state may be changed.

The claim most likely to be wrong is that the two current raw include sites
are exhaustive. The source gate and direct-header probes, rather than review,
decide that claim.

## Product attempt 1

The retained product commit is:

```text
392d3ede94293efbdcb2f164f417a3583197e90c
```

It adds `core/llfio_include.hpp`, routes both mmap headers through it, adds the
permanent direct compiler and mutation gates, and records the user-visible
diagnostic fix in `CHANGELOG.md`. No CMake manifest, target, dependency cache,
or upstream source changed.

The final product-header hashes are:

```text
llfio_include.hpp          2759489EFE683E0EB73F3ADEFD197AA7075E268223C7376A537CFB5444F59E5C
mmap_data_store.hpp        855EEA3A1AA20C883068F09FB6615185526E1D926EA7A8FA1CAB169899D78C7F
mmap_distance_matrix.hpp   3B34B4370B51B236D0DC46F5E92D23388A99841E26881D686B48ADA18D728C3A
```

## Decisive focused evidence

The permanent compiler gate uses the actual compile databases from
`build/highs-1151` and `build/nollfio`. It printed:

```text
F45_PROBE name=wrapper mode=llfio-on diagnostic=1/1 errors=1 verdict=PASS
F45_PROBE name=mmap_data_store mode=llfio-on diagnostic=1/1 errors=1 verdict=PASS
F45_PROBE name=mmap_distance_matrix mode=llfio-on diagnostic=1/1 errors=1 verdict=PASS
F45_PROBE name=llfio_off mode=llfio-off diagnostic=1/1 errors=1 verdict=PASS
F45_SOURCE_AUDIT raw_include=1/1 wrapper_routes=2/2 verdict=PASS
F45_LLFIO_DIAGNOSTIC_STATE probes=4/4 diagnostics=4/4 raw_include=1/1 wrapper_routes=2/2 skips=0 verdict=PASS
```

The mutation runner performs each compiler profile in a separate translation
unit, asserts the exact lost-diagnostic ledger, and restores exact source bytes.
It printed:

```text
F45_MUTATION_CONTROL label=initial verdict=PASS
F45_MUTATION name=missing-pop result=KILLED mutant_sha256=73BA0CDF0E6A2B2C5B2B506C2BE570365E8609788EB2500C1269D7153B625D4 restored_sha256=2759489EFE683E0EB73F3ADEFD197AA7075E268223C7376A537CFB5444F59E5C restore=pass
F45_MUTATION name=raw-data-store-include result=KILLED mutant_sha256=8ADDB9190D0CCCFBC5E1E88955D5C6556634187F1412162DEC66DDF8182595F9 restored_sha256=855EEA3A1AA20C883068F09FB6615185526E1D926EA7A8FA1CAB169899D78C7F restore=pass
F45_MUTATION name=raw-distance-matrix-include result=KILLED mutant_sha256=587C3A69ACBB21EB8F21418B9298C9B681FFA72F9A288DB9C9931DE5DC940796 restored_sha256=3B34B4370B51B236D0DC46F5E92D23388A99841E26881D686B48ADA18D728C3A restore=pass
F45_MUTATION_CONTROL label=final verdict=PASS
F45_LLFIO_DIAGNOSTIC_MUTATIONS controls=2/2 mutations=3/3 killed=3/3 survived=0 restored=3/3 skips=0 verdict=PASS
```

The missing-pop mutant retained only 1/4 diagnostics. Each raw-boundary mutant
retained exactly 3/4 and changed the source audit to raw includes 2/1 and
wrapper routes 1/2. Thus neither the source audit nor a warmed include guard can
false-green a mutation.

The unchanged F22 driver then printed the same exact ledger in both builds:

```text
F22_CPP_SILENT count=9/33 entities=Problem::readDistanceMatrix(const fs::path&),Problem::writeDistanceMatrix(const std::string&) const,Problem::writeDistanceMatrix() const,Problem::printClusters() const,Problem::writeClusters(),Problem::writeMedoidMembers(int,int) const,Problem::writeSilhouettes(),Problem::maxIter,Problem::N_repetition canonical_deprecation_lines=0
F22_CPP_DIAGNOSTICS inventory=33/33 legacy=24/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=EXPECTED_RED
```

Both F22 invocations exit 1 by that driver's registered expected-red policy.
The canonical move from 0/33 to 24/33 is the decisive F45 result; F22's exact
nine alias gaps are unchanged.

## Build and runtime matrices

Both full rebuilds exited 0. The canonical rebuild newly displayed ordinary
legacy-call warnings that the inherited LLFIO pragma had hidden; those calls
belong to F22's already-registered source-hygiene gate and are not filtered or
suppressed by F45.

An initial attempt to run both CTest matrices concurrently is invalid evidence.
Generated CTest metadata gives tests in both builds the common working
directory `C:/D/git/dtw-cpp`, and `unit_test_fileOperations` uses the relative
directory `CSV`. The collided run printed:

```text
  4/122 Test   #4: test_dense_distance_matrix_adversarial ....Exit code 0xc0000409***Exception:   5.53 sec
 91/122 Test  #91: unit_test_fileOperations ..................***Failed    5.09 sec
  remove_all: The process cannot access the file because it is being used by
  another process.: "CSV"
98% tests passed, 2 tests failed out of 122
```

The two affected tests then ran serially in each unchanged build:

```text
100% tests passed, 0 tests failed out of 2
```

The full canonical serial rerun printed:

```text
100% tests passed, 0 tests failed out of 122
Total Test time (real) =  83.97 sec
```

Its exact six capability skips were:

```text
test_cuda_correctness
test_cuda_lb_keogh
test_io_readers
test_metal_correctness
test_metal_lb_keogh
test_metal_mmap
```

The full llfio-OFF serial rerun printed:

```text
100% tests passed, 0 tests failed out of 122
Total Test time (real) =  79.60 sec
```

Its exact nine capability skips were:

```text
unit_test_mmap_data_store
unit_test_mmap_distance_matrix
test_cuda_correctness
test_cuda_lb_keogh
test_io_readers
test_metal_correctness
test_metal_lb_keogh
test_metal_mmap
unit_test_benders
```

## Verdict

**F45 CLOSED on product attempt 1 [confirmed].** All registered compiler,
source, mutation, parity, build, runtime, inventory, and skip-count bands pass.
The claim most expected to be wrong—two raw include sites being exhaustive—is
confirmed by the permanent source audit at raw include 1/1 and wrapper routes
2/2.

Rollback is local commit `392d3ed`; reverting it restores the diagnostic leak.
F22 resumes at its exhaustive C++ behavior fixture with both product attempts
still unused.
