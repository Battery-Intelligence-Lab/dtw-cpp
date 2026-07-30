# D1 — DTW recurrence and the Sakoe–Chiba adjustment window

**Verdict:** CPU and real-CUDA **CONFIRMED**; Metal source conformance
**CONFIRMED**, while real-Metal execution remains **DISCREPANCY** F12
(`[BLOCKED-ENV]` on the Windows host).

This note derives the scalar DTW objective implemented by DTWC++ and then
checks it against the live code. Sakoe and Chiba are the primary source for
the path constraints and fixed adjustment window. Their paper is not cited as
if its weighted, normalized symmetric recurrence were identical to DTWC++'s
unweighted recurrence.

## Source scope

Sakoe and Chiba state separate monotonicity and continuity conditions. Together
they imply the three-predecessor relation in equation (6); equation (7) fixes
the endpoints; and equation (8) defines the adjustment window
$|i(k)-j(k)|\le r$. Their practical calculation domain restates the window as
$j-r\le i\le j+r$. These claims were verified from the full paper scan listed
in `.claude/CITATIONS.md`.

The same paper's practical symmetric recurrence uses step-dependent weights:
in particular, a diagonal step carries twice the local cost, and the result is
normalized by $I+J$. DTWC++ instead assigns one local cost to every visited
cell, permits the three standard unit steps, and does not normalize by path
length. The fixed window is therefore directly sourced from Sakoe–Chiba; the
recurrence below is derived independently from DTWC++'s stated path objective.
The paper calls $r$ a positive integer, uses a Chebyshev vector-local
dissimilarity in its experiment, and assumes a common uniform sampling period.
DTWC++'s $w=0$ case is the natural zero-width extension of equation (8), not
a parameter value literally tested in that paper.

## Notation, units, and assumptions

Let

$$
x=(x_0,\ldots,x_{n-1}),\qquad
y=(y_0,\ldots,y_{m-1}),
$$

where $n,m\ge1$. A sample index and the band width have unit `sample`
(or timestep); conversion to physical time is valid only on a stated common
sampling grid.
Suppose scalar amplitudes have physical unit $U$. The two supported local
costs are

$$
c^{(1)}_{ij}=|x_i-y_j| \quad [U],
\qquad
c^{(2)}_{ij}=(x_i-y_j)^2 \quad [U^2].
$$

Every accumulated L1 value therefore has unit $U$; every accumulated
squared-L2 value has unit $U^2$. Neither is divided by a path length.

The derivation makes these assumptions where they are needed:

1. Inputs and local costs are finite and non-negative. This supports the
   early-abandon explanation; the recurrence itself only needs an ordered
   additive cost domain.
2. A path starts at $(0,0)$, ends at $(n-1,m-1)$, and uses steps
   $(1,0)$, $(0,1)$, or $(1,1)$. Thus it is monotone and continuous.
3. The exact-distance claims disable the API cutoff (`early_abandon < 0`).
   A cutoff request may deliberately return the no-result sentinel instead of
   the distance.
4. A mathematical accumulated result is representable below
   `numeric_limits<T>::max()`. That finite value is DTWC++'s sentinel, not an
   arithmetic infinity.
5. The band $w$ is a non-negative integer. A negative public API value is a
   dispatch convention meaning “unconstrained”, not a negative-width set.

No slope constraint, path-length normalization, diagonal weighting, or
endpoint-scaled/slanted window enters this derivation.

## Path objective

A warping path is a finite sequence

$$
P=(p_0,\ldots,p_L),\qquad p_\ell=(i_\ell,j_\ell),
$$

with $p_0=(0,0)$, $p_L=(n-1,m-1)$, and

$$
p_\ell-p_{\ell-1}\in
\{(1,0),(0,1),(1,1)\}.
$$

For either local cost $c$, the cost of $P$ is

$$
C(P)=\sum_{\ell=0}^{L} c_{i_\ell j_\ell}.
$$

If $\mathcal P_{ij}$ is the set of such paths ending at $(i,j)$, define

$$
D_{ij}=\min_{P\in\mathcal P_{ij}} C(P).
$$

The desired full DTW dissimilarity is $D_{n-1,m-1}$.

## Optimal substructure and recurrence

Every path in $\mathcal P_{ij}$, except the one-cell origin path, reaches
$(i,j)$ from exactly one of

$$
(i-1,j-1),\qquad(i-1,j),\qquad(i,j-1).
$$

Remove the last cell from any path $P$ and call the remaining prefix $Q$.
Additivity gives

$$
C(P)=C(Q)+c_{ij}.
$$

For a fixed predecessor $q$, choosing any non-optimal prefix would make
$C(Q)>D_q$; replacing it by an optimal prefix to $q$ would strictly lower
the complete path cost. Hence an optimal path to $(i,j)$ contains an
optimal prefix to its last predecessor. Conversely, appending $(i,j)$ to an
optimal path for any valid predecessor constructs a valid path to $(i,j)$.
Taking the minimum over the three exhaustive predecessor cases proves both
inequalities and therefore the equality

$$
D_{ij}
=c_{ij}+
\min\{D_{i-1,j-1},D_{i-1,j},D_{i,j-1}\}.
\tag{1}
$$

A compact boundary convention is

$$
D_{-1,-1}=0,\qquad
D_{i,-1}=+\infty,\qquad
D_{-1,j}=+\infty
$$

for $i,j\ge0$. Substitution into (1) gives

$$
D_{00}=c_{00},
$$

$$
D_{i0}=c_{i0}+D_{i-1,0}
       =\sum_{a=0}^{i}c_{a0},
\qquad
D_{0j}=c_{0j}+D_{0,j-1}
       =\sum_{b=0}^{j}c_{0b}.
$$

Thus the first row and column have exactly one admissible prefix path, and an
ascending row/column sweep evaluates every predecessor before its consumer.

## Linear-space evaluation

Equation (1) consumes only the previous column value $D_{i,j-1}$, the
previous diagonal $D_{i-1,j-1}$, and the just-updated current-column value
$D_{i-1,j}$. A buffer over the shorter axis therefore suffices:

1. before updating slot $i$, it contains $D_{i,j-1}$;
2. slot $i-1$ contains the new $D_{i-1,j}$;
3. a scalar saved before overwriting contains $D_{i-1,j-1}$.

Replacing the full matrix by this rolling state changes storage from
$O(nm)$ to $O(\min(n,m))$. It does not change the recurrence or its
arithmetic dependency graph.

## Fixed Sakoe–Chiba window

For width $w\ge0$, define the allowed cells

$$
A_w=\{(i,j):0\le i<n,\ 0\le j<m,\ |i-j|\le w\}.
$$

Let $\mathcal P_w$ be the endpoint-preserving paths whose every cell lies in
$A_w$. The constrained value is

$$
\operatorname{DTW}_w(x,y)
=\min_{P\in\mathcal P_w} C(P),
$$

with the minimum of an empty set defined as $+\infty$.

### Feasibility is necessary and sufficient

The terminal cell must be allowed, so necessity follows immediately:

$$
|(n-1)-(m-1)|=|n-m|\le w.
$$

For sufficiency, assume without loss of generality that $n\le m$. Take
$n-1$ diagonal steps from $(0,0)$ to $(n-1,n-1)$, followed by
$m-n$ horizontal steps to $(n-1,m-1)$. Along the diagonal,
$|i-j|=0$; along the tail it takes the values $1,\ldots,m-n$. If
$w\ge m-n$, every cell of this constructed path is allowed. The case
$m<n$ follows by transposition. Therefore

$$
\mathcal P_w\ne\varnothing
\quad\Longleftrightarrow\quad
w\ge|n-m|.
\tag{2}
$$

In the public API's compact notation, this is `band >= |n-m|`.

For equal lengths, $w=0$ leaves only the diagonal path. A band
$w\ge\max(n,m)-1$ covers the complete rectangle and is exactly equivalent
to an unconstrained evaluation.

### Monotonicity

If $0\le w_1\le w_2$, then

$$
A_{w_1}\subseteq A_{w_2},
\qquad
\mathcal P_{w_1}\subseteq\mathcal P_{w_2}\subseteq\mathcal P.
$$

The minimum over a subset cannot be lower than the minimum over its superset:

$$
\operatorname{DTW}_{w_1}(x,y)
\ge \operatorname{DTW}_{w_2}(x,y)
\ge \operatorname{DTW}_{\mathrm{full}}(x,y).
\tag{3}
$$

In the repository's compact contract notation,
`DTW_w(x,y) >= DTW_full(x,y)`, and the exact constrained value is
non-increasing as the band widens. Equation (3) includes an infeasible narrow
band by treating its value as $+\infty$.

## What the two “distances” mean

Both local costs are non-negative and symmetric. Transposing every cell of a
path gives a path in the opposite orientation with the same cost, so the
resulting scalar DTW values are symmetric. Nevertheless, neither accumulated
form is a metric on raw sequences.

Identity of indiscernibles already fails. For distinct sequences

$$
x=[0,1],\qquad y=[0,0,1],
$$

the path $(0,0),(0,1),(1,2)$ has zero L1 and zero squared-L2 cost.

The triangle inequality also fails for L1 local cost:

$$
x=[0,0],\quad y=[0,1],\quad z=[0,1,2],
$$

$$
\operatorname{DTW}(x,z)=3
>1+1
=\operatorname{DTW}(x,y)+\operatorname{DTW}(y,z).
$$

For squared-L2 local cost, singleton sequences suffice:

$$
x=[0],\quad y=[1],\quad z=[2],
$$

$$
\operatorname{DTW}_{\mathrm{sq}}(x,z)=4
>1+1
=\operatorname{DTW}_{\mathrm{sq}}(x,y)
+\operatorname{DTW}_{\mathrm{sq}}(y,z).
$$

The squared form is a sum of squared differences with unit $U^2$; DTWC++
does not take a final square root. Calling either result a “distance” follows
the API and DTW literature convention. Mathematically, each is a
dissimilarity, not a metric.

## Sentinel and numerical semantics

The mathematical boundary/no-path value is $+\infty$. DTWC++ deliberately
uses `numeric_limits<T>::max()` because the selected floating-point
optimizations do not assume IEEE infinity semantics. Empty input, an
infeasible window, and an exceeded early-abandon cutoff share this finite
sentinel. Consequently, `isfinite(result)` does not prove that a path ran.

No approximation is used in (1), in the rolling-buffer transformation, or in
the fixed window. Under exact arithmetic, full and rolling evaluations are
identical. For a fixed path summed sequentially in round-to-nearest arithmetic,
with no overflow or underflow, the standard model gives
$\widehat C=C(1+\delta)$, where

$$
\lvert\delta\rvert
\le\gamma_L
=\frac{L\epsilon}{1-L\epsilon}
=L\epsilon+O(L^2\epsilon^2).
$$

This is a relative bound for the path's $L$ additions; its absolute unit is
$U$ for L1 and $U^2$ for squared L2. The live build permits reassociation, and
a rounding perturbation can also select a different path at a near tie, so
this sequential bound is not claimed as a complete implementation error
bound. D17 owns the rigorous live-build and cross-precision analysis. The
decisive D1 fixtures use small integer-valued costs and exact equality, so
rounding does not enter their verdict.

## Code-conformance table

The CPU line numbers refer to derivation commit `9f78212`. The GPU rows were
reconciled after implementation commit `4583443`; their executable evidence
lives in the named F12 artifact rather than being inferred from source.

| Claim | Live implementation | Evidence and verdict |
|---|---|---|
| One local cost plus the minimum of diagonal/up/left | `dtwc/core/dtw_kernel.hpp:54-66` | `StandardCell::combine` implements (1), and `seed` implements $D_{00}=c_{00}$. **CONFIRMED**. |
| Full-matrix boundary conditions and dependency order | `dtwc/core/dtw_kernel.hpp:215-240` | Origin, first column, first row, interior sweep, and terminal cell match the derivation. **CONFIRMED** for representable allocated dimensions. |
| Linear-space state is the same recurrence | `dtwc/core/dtw_kernel.hpp:249-287` | The saved diagonal, old slot, and updated prior slot are exactly the three predecessors. **CONFIRMED**. |
| Scalar local-cost definitions and dispatch | `dtwc/warping.hpp:210-220`, `dtwc/warping.hpp:261-272` | L1 is $\lvert a-b\rvert$; SquaredL2 is $(a-b)^2$; scalar L2 correctly reduces to L1. **CONFIRMED**. |
| Short/long orientation preserves the symmetric objective | `dtwc/warping.hpp:133-150` | The shorter input becomes the first cost index; both supported scalar costs are symmetric. **CONFIRMED**. |
| Fixed allowed range $\lvert i-j\rvert\le w$ | `dtwc/core/dtw_kernel.hpp:190-207`, `dtwc/core/dtw_kernel.hpp:421-505` | Bounds are `[max(0,i-w), min(m,i+w+1))`, computed without signed overflow or `int` narrowing. **CONFIRMED**. |
| Endpoint feasibility (2) | `dtwc/core/dtw_kernel.hpp:424-436`, `dtwc/warping.hpp:389-408` | Both the shared kernel and public scalar wrapper return the finite no-path sentinel below the endpoint gap. **CONFIRMED**. |
| Negative band and full-coverage fallbacks | `dtwc/warping.hpp:394-405`, `dtwc/core/dtw_kernel.hpp:425-436` | Negative dispatches to full DTW; width at least `max_length-1` covers the complete rectangle. **CONFIRMED**. |
| Dependent and independent MV wrappers preserve feasibility | `dtwc/warping.hpp:542-561`, `dtwc/warping.hpp:598-612` | Dependent MV applies the same bound; independent MV returns one finite sentinel before channel summation. **CONFIRMED**. |
| Missing-data AROW wrapper does not bypass the fixed window | `dtwc/warping_missing_arow.hpp:153-172` | Endpoint feasibility precedes its singleton/full fallback. **CONFIRMED**. |
| Independent DP oracle and third mathematical arbiter | `tests/unit/adversarial/test_banded_dtw_adversarial.cpp:41-317` | Full-matrix DP and exhaustive path enumeration reproduce the registered costs and path counts; the live public route agrees. **CONFIRMED**: 70/70 tagged assertions. |
| Neither accumulated form is a metric | `tests/unit/adversarial/test_dtw_mathematical_properties.cpp:134-177` | Exact identity and triangle counterexamples above drive `dtwFull`. **CONFIRMED**: 11/11 tagged assertions. |
| CUDA uses the same fixed geometry | `dtwc/cuda/cuda_dtw.cu:69`, used by the pairwise kernels at `:277`, `:307`, `:455`, `:645` and one/K-vs-N kernels at `:1732`, `:1760`, `:1851`, `:1974` | One ordered-subtraction predicate implements $\lvert i-j\rvert\le w$ without signed `abs` overflow in every kernel family. The local RTX gate reproduces the independent path ledger in both singleton orientations: 515 assertions/6 F12 cases and 7,827 assertions/61 unfiltered cases. **CONFIRMED** by `.claude/baselines/2026-07-24-f12-gpu-fixed-band-parity.md`. |
| Metal fixed geometry and public sentinel | `dtwc/metal/metal_dtw.mm:137-175`, `:250-287`, `:501-530`, `:597-627`, with public normalization at `:1759` and `:2139` | Widened arithmetic clips all four fixed corridors and exact device `FLT_MAX` is translated to public `DBL_MAX`. Permanent independent-oracle cases cover pairwise and K-vs-N source routes. Three reviews found no remaining source defect, but this host has no Metal compiler/device and the binary executes zero assertions before capability skip. Source **CONFIRMED**; real-device parity remains **DISCREPANCY** F12 / `[BLOCKED-ENV]`. |

The full reference kernel currently passes its `size_t` dimensions through
`int` casts when indexing `ScratchMatrix` (`dtw_kernel.hpp:220-240`). D1 does
not claim behavior above `INT_MAX` for that allocation-heavy route. This is
**OPEN** under the R3 integer-width audit; a safe pre-allocation oversized
probe would confirm whether a public guard or wider index conversion is
required.

## Decisive artifact

The CPU preregistered ledger, inherited red, exhaustive path counts, focused
outputs, complete 114-target gate, assumptions, and rollback are recorded in
`.claude/baselines/2026-07-23-r2-d1-dtw.md`. The later GPU repair, independent
oracle, real-RTX outputs, and Metal environment probe are recorded in
`.claude/baselines/2026-07-24-f12-gpu-fixed-band-parity.md`.

**D1 final verdict:** the standard CPU recurrence, boundary conditions, scalar
cost semantics, fixed Sakoe–Chiba window, feasibility rule, and monotonicity
claim are **CONFIRMED**. CUDA geometry and exact public no-path translation are
also **CONFIRMED** on the local RTX. Metal source implements the same contract,
but its real-device executable gate remains **DISCREPANCY** F12 and cannot be
closed by Windows source inspection.
