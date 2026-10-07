# D2 — envelopes and LB_Keogh admissibility

**Verdict:** scalar envelope construction and fixed-window LB_Keogh are
**CONFIRMED** for the L1 cost, the bound TADPole uses. The proof also confirms
the `min(n,m)` prefix construction for feasible unequal-length paths under the
current fixed window. The public envelope representation is
**DISCREPANCY** F46, and TADPole's empty domain is **DISCREPANCY** F48.

## Primary-source scope

Keogh and Ratanamahatana define an upper/lower envelope and prove, in
Proposition 1, that their LB_Keogh is no larger than DTW for two sequences of
the same length under a global window `j-r <= i <= j+r`. Their point cost is
squared difference, and both their DTW and bound take a square root after the
sum. The primary identifier is DOI `10.1007/s10115-004-0154-9`; the verified
access record is in `.claude/CITATIONS.md`.

DTWC++ differs in three relevant ways:

1. its point cost is absolute difference, with no square root;
2. its symmetric bound is the maximum of two directional bounds;
3. it compares only the first `min(n,m)` rows for unequal lengths.

The paper is therefore the source for the equal-length envelope idea and its
original proposition, not a verbatim source for those three extensions. They
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

The point cost is L1:

$$
c(a,b)=\lvert a-b\rvert,
\tag{1}
$$

with unit $U$. The bound is proved for DTW under this cost only; it is not a
lower bound of a squared-L2 DTW (for $x=[0]$, $y=[\tfrac12]$ it is $\tfrac12$
against $\tfrac14$), so TADPole takes the exact route under any other metric.

The proof uses the following assumptions exactly where needed:

1. series are nonempty and contain finite real values;
2. paths start at $(0,0)$, end at $(n-1,m-1)$, and use only
   $(1,0)$, $(0,1)$, or $(1,1)$ steps;
3. the actual DTW window is the fixed set
   $A_w=\{(i,j):\lvert i-j\rvert\le w\}$ for a nonnegative integer $w$;
4. the envelope radius $r$ covers the DTW radius: `r >= w`;
5. local costs are unweighted, additive, and nonnegative;
6. arithmetic is exact and accumulated values are representable.

Missing-value policies, normalized/path-averaged objectives, derivative
series, ADTW penalties, Soft-DTW soft minima, and floating-point last-ULP
behavior are not smuggled into these assumptions. Their validity must be
established separately.

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

For a set $S$ of query rows, the directional bound is

$$
\operatorname{LB}_{x\mid y,r}(S)
=
\sum_{i\in S}
\delta\!\left(x_i;L_i^{(r)}(y),U_i^{(r)}(y)\right).
\tag{8}
$$

Equation (8) has the unit of its DTW objective, $U$.

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
$y_{j_i}\in[L_i^{(r)}(y),U_i^{(r)}(y)]$. Applying (7) row by row gives

$$
\delta\!\left(x_i;L_i^{(r)}(y),U_i^{(r)}(y)\right)
\le c(x_i,y_{j_i}).
\tag{9}
$$

Summing the chosen distinct cells, then using nonnegativity for all unchosen
path cells,

$$
\begin{aligned}
\operatorname{LB}_{x\mid y,r}(S)
&\le
\sum_{i\in S}c(x_i,y_{j_i})\\
&\le
\sum_{(a,b)\in P}c(x_a,y_b)
=C(P).
\end{aligned}
\tag{10}
$$

This holds for every admissible $P$, hence it holds for the minimum:

$$
\operatorname{LB}_{x\mid y,r}(S)
\le \operatorname{DTW}_w(x,y).
\tag{11}
$$

No property of rows outside $S$ was used. This observation is the entire
unequal-prefix extension.

## Reverse and symmetric bounds

Transpose the path. Every candidate row is then visited, and the identical
argument proves

$$
\operatorname{LB}_{y\mid x,r}(T)
\le \operatorname{DTW}_w(x,y)
\tag{12}
$$

for any included candidate-row subset $T$.

If two numbers are each no larger than the same quantity, their maximum is
also no larger:

$$
\operatorname{LB}_{\mathrm{sym}}
=
\max\!\left(
\operatorname{LB}_{x\mid y,r},
\operatorname{LB}_{y\mid x,r}
\right)
\le \operatorname{DTW}_w(x,y).
\tag{13}
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
\tag{14}
$$

Equations (11)–(13) immediately prove admissibility of forward, reverse, and
symmetric prefix bounds. Truncation discards nonnegative terms; it does not
invent a new charge.

This theorem has three load-bearing qualifications:

1. a finite path requires `w >= |n-m|`;
2. each envelope still requires `r >= w`;
3. the geometry is the current fixed condition $\lvert i-j\rvert\le w$.

It does not transfer automatically to a slope-scaled, asymmetric, or
data-dependent window. Such a window would require donor sets derived from
its own cell geometry.

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
\tag{15}
$$

A forward pass maintains decreasing and increasing deques for the trailing
maximum and minimum. A backward pass does the same for the leading interval,
then combines maxima with `max` and minima with `min`. Every index is inserted
once and removed at most once from each deque, so time is $O(m)$ and temporary
storage is $O(m+r)$ in the current implementation.

Equation (15) is an exact set decomposition. The deque removes only an older
value dominated by a newer value that remains in the window for at least as
long. Therefore it preserves the extrema exactly.

No approximation is used in envelope construction or in the bound. The
finite oracle uses exactly representable values. Floating reductions can
round in a different order from DTW accumulation; the analytic and
cross-precision threshold band is not derived here.

## The full-DTW call site

TADPole is the one caller. It can skip a pair because its density
stage needs only a threshold decision. For supported finite, nonempty,
equal-length, univariate Standard-L1 data whose series length is representable
by the integer band API, it replaces a negative band by the series length and
therefore builds the global envelope in (5). F46's radius-contract audit
includes the unchecked `size_t`-to-`int` narrowing at this call site. An exact
distance matrix needs every pair and uses no bound.

The permanent test distinguishes a disabled TADPole LB, the unsafe radius-zero
envelope, and the intended global envelope with two orthogonal fixtures. Their
registered `pruned_by_lb` fingerprint is `(0,1)`: no false prune for a
zero-DTW warped pair, and one real bound decision for a separated-range pair.

This confirmation does not cover empty series. F48 records the independent
case where an empty diagonal upper bound of zero disagrees with the exact
no-path sentinel.

## The negative band

`compute_envelopes` reads a negative band as full DTW and builds (5). A
narrower envelope can exceed full DTW. For

```text
x = [0,0,0,0,1,1,1,1,1,1]
y = [0,0,0,0,0,0,1,1,1,1]
```

full L1 DTW is zero, while the symmetric L1 bounds are 2 at radius zero, 1 at
radius one, and 0 for the global envelope. The D2 gate pins `-1` to the global
envelope and the bound 0.

The mutable `Envelope` type still does not record its window: valid-shaped
arrays can come from an unrelated or too-narrow window. F46 keeps an explicit
full/radius descriptor, shape/coverage validation, and alias safety.

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
| TADPole full-DTW call site | a zero-DTW warped pair and a separated pair | 2 | `pruned_by_lb` `(0,1)` |

Every admissibility case checks forward, reverse, and symmetric L1 bounds
against the explicit minimum path cost; the test REQUIREs each case count, so
a run that skipped the enumeration fails.

The non-degenerate direction ledger is:

| Bound | Forward | Reverse | Symmetric max |
|---|---:|---:|---:|
| L1 (`U`) | 8 | 2 | 8 |

The preregistered bands and the first verbatim run are in
`.claude/baselines/2026-07-30-d2-lb-keogh.md`.

## Code-conformance table

| Claim | Live code | Verdict |
|---|---|---|
| Fixed CPU cell geometry and feasibility | `dtw_band_bounds` and `dtw_kernel_banded` in `dtwc/core/dtw_kernel.hpp` | **CONFIRMED** by D1 |
| Centered scalar envelope, equations (2)–(5) | `compute_envelopes` in `dtwc/core/lower_bound_impl.hpp` | **CONFIRMED**; a negative band builds the global envelope (5) |
| L1 projection sum, equations (6)–(11) | pointer `lb_keogh` in `dtwc/core/lower_bound_impl.hpp` | **CONFIRMED** |
| Symmetric maximum, equation (13) | `lb_keogh_symmetric` in `dtwc/core/lower_bound_impl.hpp` | **CONFIRMED** |
| Prefix truncation, equation (14) | `Envelope` `lb_keogh` in `dtwc/core/lower_bound_impl.hpp` | Math **CONFIRMED** for feasible fixed windows |
| TADPole global-envelope conversion | `bounds_valid` and `tadpole` in `dtwc/algorithms/tadpole.cpp` | **CONFIRMED** for finite, nonempty, equal-length Standard-L1 with integer-representable lengths; empty case is F48 and radius narrowing is F46 |
| Exhaustive independent oracle | `tests/unit/core/test_lb_keogh_derivation.cpp` | **CONFIRMED**, non-skippable |
| Public envelope shape/window contract | `Envelope`, `envelope_covers` and the `lb_keogh` overloads in `dtwc/core/lower_bound_impl.hpp` | **DISCREPANCY** F46: unchecked read/truncation and no provenance |

## Scope verdicts

- **CONFIRMED:** the scalar algebra, direct production formulas, envelope
  monotonicity, full/global construction, symmetric maximum, and feasible
  unequal fixed-window prefix theorem.
- **CONFIRMED:** the nonempty TADPole full-DTW call site executes with the
  registered safety/reachability fingerprint.
- **DISCREPANCY:** the F46 and F48 subjects named in the table. Neither is
  hidden by the green scalar oracle.
- **OPEN:** floating-point threshold safety, multivariate series, and all
  non-Standard objectives.

The claim most expected to need refinement is bit-level threshold
admissibility. The proof is exact-arithmetic; a bound and DTW accumulated in
different orders can straddle the same floating cutoff by an ulp. That guard
must be derived before a universal floating-point pruning claim is made.
