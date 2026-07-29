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
