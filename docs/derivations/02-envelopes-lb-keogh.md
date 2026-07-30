# D2 — envelopes and LB_Keogh admissibility

**Verdict:** scalar CPU envelope construction and fixed-window LB_Keogh are
**CONFIRMED** for L1 and unrooted squared-L2 costs. The proof also confirms
the `min(n,m)` prefix construction for feasible unequal-length paths under the
current fixed window. The public envelope representation is
**DISCREPANCY** F46, squared LB_Kim is **DISCREPANCY** F47, TADPole's empty
domain is **DISCREPANCY** F48, direct-call band/cache provenance is
**DISCREPANCY** F49, and GPU execution still has the F27–F30/F50
qualifications mapped below.

## Primary-source scope

Keogh and Ratanamahatana define an upper/lower envelope and prove, in
Proposition 1, that their LB_Keogh is no larger than DTW for two sequences of
the same length under a global window `j-r <= i <= j+r`. Their point cost is
squared difference, and both their DTW and bound take a square root after the
sum. The primary identifier is DOI `10.1007/s10115-004-0154-9`; the verified
access record is in `.claude/CITATIONS.md`.

DTWC++ differs in four relevant ways:

1. its default point cost is absolute difference;
2. its squared-L2 result is the unrooted sum;
3. its symmetric bound is the maximum of two directional bounds;
4. its GPU kernels compare only the first `min(n,m)` rows for unequal lengths.

The paper is therefore the source for the equal-length envelope idea and its
original proposition, not a verbatim source for those four extensions. They
are derived here.

Lemire's 2006 streaming maximum-minimum filter is the source for the
monotone-deque technique used to construct extrema in linear time. It is not a
source for DTW admissibility; its primary preprint identifier is arXiv
`cs/0610046`. Sakoe and Chiba's fixed-window geometry was derived and checked
separately in D1.

## Notation, units, and assumptions

Let

$$
x=(x_0,\ldots,x_{n-1}),\qquad
y=(y_0,\ldots,y_{m-1}),
$$

with `n,m >= 1`. A sample index, DTW radius, and envelope radius have unit
`sample`. Scalar amplitudes have an arbitrary physical unit $U$.

For $p\in\{1,2\}$ define the point cost

$$
c_p(a,b)=\lvert a-b\rvert^p.
\tag{1}
$$

Thus $p=1$ is L1 and has unit $U$; $p=2$ is unrooted squared L2 and has unit
$U^2$. DTWC++ does not take a final square root.

The proof uses the following assumptions exactly where needed:

1. series are nonempty and contain finite real values;
2. paths start at $(0,0)$, end at $(n-1,m-1)$, and use only
   $(1,0)$, $(0,1)$, or $(1,1)$ steps;
3. the actual DTW window is the fixed set
   $A_w=\{(i,j):\lvert i-j\rvert\le w\}$ for a nonnegative integer $w$;
4. the envelope radius $r$ covers the DTW radius: `r >= w`;
5. local costs are unweighted, additive, and nonnegative;
6. arithmetic is exact and accumulated values are representable.
7. for the multivariate extension, channels share a commensurate unit $U$
   after any required scaling or nondimensionalization.

Missing-value policies, normalized/path-averaged objectives, derivative
series, ADTW penalties, Soft-DTW soft minima, GPU threshold casting, and
floating-point last-ULP behavior are not smuggled into these assumptions.
Their validity must be established separately.

For unequal lengths, a finite fixed-window path exists if and only if
`w >= |n-m|`. A finite lower bound below DTWC++'s no-path sentinel when this
condition fails is only vacuously admissible; it is not evidence about a
finite unequal-length alignment.

## Centered envelopes

For a candidate series $y$ and radius $r\ge0$, define the clipped donor-index
set for row $i$ by

$$
J_i^{(r)}(y)
=
\{j:0\le j<m,\ \lvert i-j\rvert\le r\}.
\tag{2}
$$

For every row used in the bound, this set is nonempty under the assumptions
above. Its lower and upper envelopes are

$$
L_i^{(r)}(y)=\min_{j\in J_i^{(r)}(y)}y_j,\qquad
U_i^{(r)}(y)=\max_{j\in J_i^{(r)}(y)}y_j.
\tag{3}
$$

Equivalently, the implementation scans the inclusive index interval

```text
[max(0,i-r), min(m-1,i+r)].
```

If $r_2\ge r_1$, then
$J_i^{(r_1)}\subseteq J_i^{(r_2)}$, so

$$
L_i^{(r_2)}\le L_i^{(r_1)}
\le U_i^{(r_1)}\le U_i^{(r_2)}.
\tag{4}
$$

A wider envelope can only weaken the bound. Equality between the DTW radius
and envelope radius is not required; coverage `r >= w` is sufficient.

For full DTW every candidate index may align with every included query row.
The correct envelope therefore repeats the global extrema:

$$
L_i^{(\mathrm{full})}=\min_j y_j,\qquad
U_i^{(\mathrm{full})}=\max_j y_j.
\tag{5}
$$

For the equal-length and `min(n,m)`-prefix forms used here, radius $m-1$ is
mathematically sufficient for a candidate of length $m$; passing $m$ is also
safe and selects DTWC++'s explicit global-extrema fast path. This statement
does not license an all-row bound for a longer query without checking that
every included row's donor set is covered.

## Projection onto an envelope interval

For a closed interval $[L,U]$ with $L\le U$, define the scalar projection
excess

$$
\delta(a;L,U)
=
\begin{cases}
L-a,&a<L,\\
0,&L\le a\le U,\\
a-U,&a>U.
\end{cases}
\tag{6}
$$

This is the distance from $a$ to the interval. For every $b\in[L,U]$,

$$
\delta(a;L,U)\le\lvert a-b\rvert.
\tag{7}
$$

The three cases prove (7) directly:

- if $a<L$, then $b\ge L$, hence $b-a\ge L-a$;
- if $a>U$, then $b\le U$, hence $a-b\ge a-U$;
- if $a\in[L,U]$, the left side is zero.

Both sides are nonnegative, so the monotonicity of $t\mapsto t^p$ for
$p\in\{1,2\}$ gives

$$
\delta(a;L,U)^p\le\lvert a-b\rvert^p.
\tag{8}
$$

For a set $S$ of query rows, the directional bound is

$$
\operatorname{LB}^{(p)}_{x\mid y,r}(S)
=
\sum_{i\in S}
\delta\!\left(x_i;L_i^{(r)}(y),U_i^{(r)}(y)\right)^p.
\tag{9}
$$

Equation (9) has the same units as its DTW objective: $U$ for $p=1$ and
$U^2$ for $p=2$.

## One-directional admissibility

Let $P$ be any path admitted by the actual radius $w$. Because row increments
are zero or one, a path from row 0 to row $n-1$ visits every row. For each
$i\in S$, choose one visited cell $(i,j_i)$.

The selected cells are distinct because their row indices are distinct.
Moreover,

$$
\lvert i-j_i\rvert\le w\le r,
$$

so $j_i\in J_i^{(r)}(y)$ and
$y_{j_i}\in[L_i^{(r)}(y),U_i^{(r)}(y)]$. Applying (8) row by row gives

$$
\delta\!\left(x_i;L_i^{(r)}(y),U_i^{(r)}(y)\right)^p
\le c_p(x_i,y_{j_i}).
\tag{10}
$$

Summing the chosen distinct cells, then using nonnegativity for all unchosen
path cells,

$$
\begin{aligned}
\operatorname{LB}^{(p)}_{x\mid y,r}(S)
&\le
\sum_{i\in S}c_p(x_i,y_{j_i})\\
&\le
\sum_{(a,b)\in P}c_p(x_a,y_b)
=C_p(P).
\end{aligned}
\tag{11}
$$

This holds for every admissible $P$, hence it holds for the minimum:

$$
\operatorname{LB}^{(p)}_{x\mid y,r}(S)
\le \operatorname{DTW}^{(p)}_w(x,y).
\tag{12}
$$

No property of rows outside $S$ was used. This observation is the entire
unequal-prefix extension.

## Reverse and symmetric bounds

Transpose the path. Every candidate row is then visited, and the identical
argument proves

$$
\operatorname{LB}^{(p)}_{y\mid x,r}(T)
\le \operatorname{DTW}^{(p)}_w(x,y)
\tag{13}
$$

for any included candidate-row subset $T$.

If two numbers are each no larger than the same quantity, their maximum is
also no larger:

$$
\operatorname{LB}^{(p)}_{\mathrm{sym}}
=
\max\!\left(
\operatorname{LB}^{(p)}_{x\mid y,r},
\operatorname{LB}^{(p)}_{y\mid x,r}
\right)
\le \operatorname{DTW}^{(p)}_w(x,y).
\tag{14}
$$

Their sum is not generally admissible because the two proofs may charge the
same path cell twice. For singleton series $x=[0]$, $y=[1]$, each direction
equals DTW, while their sum is twice DTW. The implementation correctly takes
the maximum.

## Unequal-length prefix theorem

Let

$$
k=\min(n,m),\qquad
S=T=\{0,\ldots,k-1\}.
\tag{15}
$$

Equations (12)–(14) immediately prove admissibility of forward, reverse, and
symmetric prefix bounds. Truncation discards nonnegative terms; it does not
invent a new charge.

This theorem has three load-bearing qualifications:

1. a finite path requires `w >= |n-m|`;
2. each envelope still requires `r >= w`;
3. the geometry is the current fixed condition $\lvert i-j\rvert\le w$.

It does not transfer automatically to a slope-scaled, asymmetric, or
data-dependent window. Such a window would require donor sets derived from
its own cell geometry.

The original F29 fixture used series of lengths two and three at band zero.
Under the repaired fixed geometry it has no path, so its claimed zero DTW is
not a valid counterexample. At the minimally feasible band one, the path
$(0,0),(0,1),(1,2)$ has zero cost and both prefixes have zero bound. F29 now
owns executable CUDA/Metal confirmation rather than a mathematical repair.

## Multivariate dependent and independent extensions

For a $d$-channel dependent DTW, construct (3) independently per channel.
Here the common-unit assumption above is load-bearing: unweighted addition is
the live objective only after channels are expressed in a shared commensurate
unit or nondimensionalized scale. Raw heterogeneous physical units require an
explicit scaling/weighting model; without one, the sum has no single physical
unit and the $U$/$U^2$ ledger below is not defined.

At row $i$, these intervals form an axis-aligned box. For the additive
multivariate L1 and squared-L2 point costs,

$$
c_{p,\mathrm{MV}}(a,b)
=
\sum_{\ell=0}^{d-1}\lvert a_\ell-b_\ell\rvert^p.
\tag{16}
$$

Every aligned candidate vector lies inside the box coordinate by coordinate.
Applying (8) to each coordinate and summing proves that the box-projection
cost is no larger than the local multivariate path-cell cost. Equation (11)
then applies unchanged.

This proves the low-level `lb_keogh_mv` and `lb_keogh_mv_squared` formulas for
dependent multivariate DTW with additive L1 or squared-L2 point costs.

Independent DTW allows each channel $\ell$ to choose its own path
$P_\ell$. Apply the scalar proof (12) separately to every channel, then sum:

$$
\sum_{\ell=0}^{d-1}\operatorname{LB}^{(p)}_\ell
\le
\sum_{\ell=0}^{d-1}\min_{P_\ell}C_{p,\ell}(P_\ell)
=
\operatorname{DTW}^{(p)}_{\mathrm I}.
\tag{17}
$$

Different paths do not invalidate (17); they are precisely why the minimum is
taken before the channel sum. Thus the same production formula is also a
lower bound for independent DTW when its objective is the additive sum of
per-channel L1 or squared-L2 DTWs and each envelope covers that channel's
fixed window.

The opposite-warp discriminator uses query channels `[0,3,3]` and `[0,0,2]`
against candidate channels `[0,0,2]` and `[0,3,3]` at radius one. Recursive
scalar path enumeration gives independent L1 and squared objectives of
`4 U` and `4 U^2`; the production multivariate bounds are `3 U` and `3 U^2`.
The shared-path dependent objectives are instead `8 U` and `20 U^2`, so they
cannot masquerade as the independent arbiter.

Neither proof establishes a sum-of-coordinate excess bound for multivariate
Euclidean `MVL2Dist`, whose square root couples channels. The low-level
primitives are not wired into `Problem`'s automatic matrix route.

## Linear-time envelope construction

A direct implementation of (3) scans up to $2r+1$ values per output. Including
the mandatory read/write at radius zero, its work is
$\Theta(m\min(m,2r+1))$, or $O(m(r+1))$. The production implementation uses
monotone index deques.

For each position $i$, split the centered interval into a trailing and a
leading interval:

$$
[i-r,i+r]\cap[0,m-1]
=
\bigl([i-r,i]\cap[0,m-1]\bigr)
\cup
\bigl([i,i+r]\cap[0,m-1]\bigr).
\tag{18}
$$

A forward pass maintains decreasing and increasing deques for the trailing
maximum and minimum. A backward pass does the same for the leading interval,
then combines maxima with `max` and minima with `min`. Every index is inserted
once and removed at most once from each deque, so time is $O(m)$ and temporary
storage is $O(m+r)$ in the current implementation.

Equation (18) is an exact set decomposition. The deque removes only an older
value dominated by a newer value that remains in the window for at least as
long. Therefore it preserves the extrema exactly.

No approximation is used in envelope construction or in the bound. The
finite oracle uses exactly representable values. Floating reductions can
round in a different order from DTW accumulation; the analytic and
cross-precision threshold band belongs to D17.

## Full-DTW call sites

Two CPU call-site classes currently handle a negative DTW band differently:

1. `fill_distance_matrix_pruned` disables envelope-based bounds when
   `band < 0`. It may still use L1 LB_Kim under the default cascade. Since an
   exact matrix needs every final value, an early-abandoned pair is retried;
   this route does not skip exact pair computation. D2 does not prove LB_Kim,
   and F47 records its squared-cost trait mismatch.
2. TADPole can skip a pair because its density stage needs only a threshold
   decision. For supported finite, nonempty, equal-length, univariate
   Standard-L1 data whose series length is representable by the integer band
   API, it replaces a negative band by the series length and therefore builds
   the global envelope in (5). F46's radius-contract audit includes the
   unchecked `size_t`-to-`int` narrowing at this call site.

The permanent test distinguishes a disabled TADPole LB, the unsafe radius-zero
envelope, and the intended global envelope with two orthogonal fixtures. Their
registered `pruned_by_lb` fingerprint is `(0,1)`: no false prune for a
zero-DTW warped pair, and one real bound decision for a separated-range pair.

This confirmation does not cover empty series. F48 records the independent
case where an empty diagonal upper bound of zero disagrees with the exact
no-path sentinel. F49 separately records the direct matrix function's ability
to receive a band that disagrees with `Problem::band`.

## The negative-band discrepancy

The low-level helper currently normalizes its integer argument as

```cpp
const std::size_t w =
  static_cast<std::size_t>(std::max(band, 0));
```

Consequently `compute_envelopes(series,-1,...)` constructs the radius-zero
identity envelope, not (5). For

```text
x = [0,0,0,0,1,1,1,1,1,1]
y = [0,0,0,0,0,0,1,1,1,1]
```

full L1 DTW is zero, while the symmetric L1 bounds are 2 at radius zero, 1 at
radius one, and 0 for the global envelope. Passing `-1` to the helper returns
the invalid value 2 against full DTW 0.

The two audited production callers avoid that exact misuse, but the installed
helper and mutable `Envelope` type do not encode the distinction. Valid-shaped
arrays can also come from an unrelated or too-narrow window. F46 owns an
explicit full/radius descriptor, shape/coverage validation, and alias safety.

## Executable oracle

The decisive test uses two independent mathematical constructions:

- direct $\Theta(m\min(m,2r+1))$ extrema scans, not the production deques;
- recursive enumeration of every admissible monotone path, not a DTW
  recurrence or rolling buffer.

Its exact inventories are:

| Subject | Alphabet and dimensions | Cases | Verdict |
|---|---|---:|---|
| Envelope equality | `{-2,0,3}`, lengths 1–5, radii 0–n | 2,004 | zero mismatches |
| Equal-length admissibility | `{-1,0,2}`, lengths 1–4, all ordered pairs and radii | 28,602 | zero violations |
| Feasible unequal prefixes | `{-1,0,2}`, lengths 1–4, all unequal ordered pairs and feasible radii | 17,712 | zero violations |
| Independent-MV discriminator | two channels with opposite preferred warp directions | 1 | bound `3/3`, independent DTW `4/4`, dependent DTW `8/20` |
| Full-DTW caller classes | exact matrix and TADPole | 2/2 | both reached |

Every admissibility case checks forward, reverse, and symmetric bounds for
both $p=1$ and $p=2$ against the explicit minimum path cost.

The non-degenerate direction/metric ledger is:

| Bound | Forward | Reverse | Symmetric max |
|---|---:|---:|---:|
| L1 (`U`) | 8 | 2 | 8 |
| squared (`U^2`) | 22 | 4 | 22 |

The singleton metric discriminator is L1 `0.5 U` versus squared `0.25 U^2`.
The strengthened focused executable prints:

```text
D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS
All tests passed (65 assertions in 1 test case)
```

The full verbatim run, preregistered bands, and both execution seeds are in
`.claude/baselines/2026-07-30-d2-lb-keogh.md`.

## Code-conformance table

| Claim | Live code | Verdict |
|---|---|---|
| Scalar L1 and squared point costs, equation (1) | `dtwc/core/distance_metric.hpp:20-43` | **CONFIRMED** |
| Fixed CPU cell geometry and feasibility | `dtwc/core/dtw_kernel.hpp:190-205,420-436` | **CONFIRMED** by D1 |
| Centered scalar envelope, equations (2)–(5) | `dtwc/core/lower_bound_impl.hpp:58-126` | **CONFIRMED** for `band>=0`; negative coercion is **DISCREPANCY** F46 |
| L1 projection sum, equations (6)–(12) | `dtwc/core/lower_bound_impl.hpp:181-203` | **CONFIRMED** |
| Squared projection sum | `dtwc/core/lower_bound_impl.hpp:551-570` | **CONFIRMED** |
| Symmetric maximum, equation (14) | `dtwc/core/lower_bound_impl.hpp:371-388` | **CONFIRMED** for its L1 wrapper; no public squared symmetric wrapper |
| Prefix truncation, equation (15) | `dtwc/core/lower_bound_impl.hpp:356-368`; CUDA `dtwc/cuda/cuda_dtw.cu:782-813`; Metal `dtwc/metal/metal_dtw.mm:932-954` | Math **CONFIRMED** for feasible fixed windows; real backends remain F29 |
| Per-channel dependent/independent MV extensions, equations (16)–(17) | `dtwc/core/lower_bound_impl.hpp:419-527,593-611`; `tests/unit/adversarial/test_lb_keogh_derivation.cpp:364-408` | **CONFIRMED** for additive L1/squared costs; low-level only |
| Exact-matrix full-window guard | `dtwc/core/pruned_distance_matrix.cpp:128-150,216-238,339-378` | **CONFIRMED**; Keogh disabled for negative bands and unequal lengths |
| TADPole global-envelope conversion | `dtwc/algorithms/tadpole.cpp:149-160,178-190,219-224` | **CONFIRMED** for finite, nonempty, equal-length Standard-L1 with integer-representable lengths; empty case is F48 and radius narrowing is F46 |
| Exhaustive independent oracle | `tests/unit/adversarial/test_lb_keogh_derivation.cpp:127-526`; `tests/CMakeLists.txt:128-144` | **CONFIRMED**, non-skippable |
| Public envelope shape/window contract | `dtwc/core/lower_bound_impl.hpp:214-223,326-388` | **DISCREPANCY** F46: unchecked read/truncation and no provenance |
| Squared LB_Kim compatibility | `dtwc/core/lower_bounds.hpp:42-52`; `dtwc/core/lower_bound_impl.hpp:225-324` | **DISCREPANCY** F47: L1-unit result advertised for squared DTW |
| CUDA/Metal squared pruning | CUDA `dtwc/cuda/cuda_dtw.cu:792-810`; Metal `dtwc/metal/metal_dtw.mm:946-954` | **DISCREPANCY** F27: both sum L1 excess |
| Metal full-DTW envelope choice | `dtwc/metal/metal_dtw.mm:1512-1515` | **DISCREPANCY** F28: default/explicit radius may be too narrow |
| Explicit GPU LB requests | CUDA `dtwc/cuda/cuda_dtw.cu:1476-1480`; Metal `dtwc/metal/metal_dtw.mm:1493-1503,1536-1550` | **DISCREPANCY** F30: requests can silently disable or fall back |
| Extreme GPU radius arithmetic | CUDA `dtwc/cuda/cuda_dtw.cu:733-744`; Metal `dtwc/metal/metal_dtw.mm:890-901` | **DISCREPANCY** F50 by source; numeric device result still inferred |

## Scope verdicts

- **CONFIRMED:** the scalar algebra, direct production formulas, envelope
  monotonicity, full/global construction, symmetric maximum, and feasible
  unequal fixed-window prefix theorem.
- **CONFIRMED:** the two nonempty CPU full-DTW call-site policies execute with
  the registered safety/reachability fingerprints.
- **CONFIRMED:** dependent multivariate additive L1/squared formulas follow by
  coordinatewise projection; independent additive DTW follows by summing the
  separately minimized scalar bounds. Both remain low-level primitives.
- **FALSIFIED:** the old F29 claim that prefix truncation itself is
  inadmissible under the current fixed window. Real device conformance is
  still an open F29 gate.
- **DISCREPANCY:** F46–F50 and the pre-existing F27–F30 subjects named in the
  table. None is hidden by the green scalar oracle.
- **OPEN:** floating-point threshold safety, GPU FP32 reduction/casting,
  multivariate Euclidean bounds, and all non-Standard objectives.

The claim most expected to need refinement is bit-level threshold
admissibility. The proof is exact-arithmetic; a bound and DTW accumulated in
different orders can straddle the same floating cutoff by an ulp. D17 must
derive that guard before a universal floating-point pruning claim is made.
