# D3 — LB_Enhanced and the local LB_Webb_NoLR bound

**Verdict:** LB_Enhanced and DTWC++'s local
LB_Webb_NoLR-plus-tail-cap implementation are **CONFIRMED** lower bounds for
finite, nonempty, equal-length scalar series under L1 and unrooted
squared-L2 point costs, provided the envelopes, lower bound, and DTW use the
same saturated fixed window. Directional Enhanced with effective `V=1`
dominates matching-direction LB_Keogh; at effective `V>=2` neither Enhanced
nor Keogh dominates. The local directional Webb result dominates
matching-direction Keogh, and its symmetric maximum dominates symmetric
Keogh. No ordering is claimed between the local NoLR variant and full Webb
Algorithm 2.

F54's live Enhanced/Keogh take-maximum route and F57's extreme-radius CPU
arithmetic are **CONFIRMED** by the focused executable evidence below. The
public envelope shape/provenance contract remains **DISCREPANCY** F46, GPU
extreme-radius arithmetic remains F50, and floating threshold identity
remains D17.

## Primary-source scope and implementation identity

Tan, Petitjean, and Webb, *Elastic bands across the path: A new framework and
method to lower bound DTW*, SDM 2019, DOI
`10.1137/1.9781611975673.59`, provide the elastic-cut construction in
Theorem 3.1, the LB_Enhanced formula in Eq. 3.7, and its admissibility result
in Theorem 3.2. The paper explicitly describes `LB_Enhanced^1` as uniformly
tighter than LB_Keogh. The effective-`V>=2` no-ordering result in this
repository comes from exact witnesses, not from that paper.

Webb and Petitjean, *Tight lower bounds for Dynamic Time Warping*, *Pattern
Recognition* 115 (2021) 107895, DOI
`10.1016/j.patcog.2021.107895`, give the four-point correction condition in
Theorem 2 and the bridge/correction formulas in Eqs. 26–43. Full Algorithm 2
contains `MinLRPaths`. DTWC++'s historically named `lb_webb` function instead
matches the later all-index `LB_Webb_NoLR` formula and adds a conservative
trailing-flag cap.

These variants do not have a universal ordering. The paper's Wafer table
reports average tightness 0.96904 for NoLR and 0.96891 for full Webb. That
reversal rules out the former claim that omitting `MinLRPaths` can only
loosen the result. The one-sided result proved here applies only to DTWC++'s
separate trailing-flag cap:

$$
\operatorname{Webb}_{\mathrm{cap}}
\le
\operatorname{Webb}_{\mathrm{NoLR}}
\le
\operatorname{DTW}.
\tag{1}
$$

The verified primary-source records are in `.claude/CITATIONS.md`. No
licensed reference implementation is copied.

## Assumptions, units, and exclusions

Let

$$
A=(A_0,\ldots,A_{n-1}),\qquad
B=(B_0,\ldots,B_{n-1}),
\qquad n\ge1.
\tag{2}
$$

Amplitudes have arbitrary physical unit $U$. For
$p\in\{1,2\}$, define

$$
c_p(a,b)=\lvert a-b\rvert^p.
\tag{3}
$$

For $p=1$, a point cost and accumulated DTW have unit $U$. For $p=2$,
they have unit $U^2$. The squared form is an additive point cost, not a
metric, and DTWC++ does not take a final square root.

For an integer API radius `band`, the common effective radius is

$$
w=\min\!\left(\max(\mathrm{band},0),n-1\right).
\tag{4}
$$

Thus a negative helper radius means radius zero, not full DTW. Every radius
at least $n-1$ has identical global-window geometry.

The proofs use these assumptions where they first become necessary:

1. $A$ and $B$ are finite scalar real series with the same nonzero length.
2. A path starts at $(0,0)$, ends at $(n-1,n-1)$, uses only
   $(1,0)$, $(0,1)$, and $(1,1)$ steps, and obeys
   $\lvert i-j\rvert\le w$.
3. Envelopes, bounds, and the bounded DTW use the same saturated window,
   source series, and point cost.
4. Point costs are the nonnegative additive costs in (3).
5. Arithmetic is exact and every accumulated value is representable.
6. For Enhanced with $n>1$, the effective number of bands is

   $$
   v=\min\!\left(\max(V,1),\left\lfloor n/2\right\rfloor\right).
   \tag{5}
   $$

NaN/Inf, unequal lengths, mutable or mismatched envelope provenance, aliasing,
lengths not representable by current low-level indices, multivariate data,
rooted squared distance, normalized or path-averaged DTW, derivative and
weighted variants, ADTW, and Soft-DTW are not hidden inside these assumptions.
F46 owns the public envelope and length boundary; D17 owns floating
last-ULP threshold decisions.

## Clipped envelopes and interval projection

For any series $S$, define its clipped donor window and primary envelopes by

$$
I_i=\{j:0\le j<n,\ \lvert i-j\rvert\le w\},
\tag{6}
$$

$$
U_i^S=\max_{j\in I_i}S_j,\qquad
L_i^S=\min_{j\in I_i}S_j.
\tag{7}
$$

The secondary envelopes used by NoLR are

$$
LU^S=L_w(U_w(S)),\qquad
UL^S=U_w(L_w(S)).
\tag{8}
$$

The names match the operation order: `lu` is a lower envelope of the upper
envelope, while `ul` is an upper envelope of the lower envelope.

For an interval $[L,U]$, define its projection excess

$$
d(a;L,U)=
\begin{cases}
L-a,&a<L,\\
0,&L\le a\le U,\\
a-U,&a>U.
\end{cases}
\tag{9}
$$

For every $b\in[L,U]$ and $p\in\{1,2\}$,

$$
d(a;L,U)^p\le\lvert a-b\rvert^p=c_p(a,b).
\tag{10}
$$

This follows because $d$ is the smallest absolute displacement from $a$ to
the interval, and $t\mapsto t^p$ is monotone for $t\ge0$. The middle
Enhanced term is this continuous interval projection. It is not the minimum
over the discrete donor samples; if $a$ lies between two extrema, its
projection is zero even when no donor equals $a$.

## Enhanced elastic-cut sets

Assume first that $n>1$. For every $0\le k<v$, define the left elastic cut

$$
\mathcal L_k
=
\{(k,k)\}
\mathbin{\cup}
\{(k,j),(j,k):
\max(0,k-w)\le j<k\}.
\tag{11}
$$

Reflect both coordinates through $n-1$ to obtain the right cut

$$
\mathcal R_k
=
\{(n-1-i,n-1-j):(i,j)\in\mathcal L_k\}.
\tag{12}
$$

For every middle row $v\le i<n-v$, define

$$
\mathcal M_i
=
\{(i,j):\max(0,i-w)\le j\le\min(n-1,i+w)\}.
\tag{13}
$$

### Crossing lemma

Every admissible path crosses every set in (11)–(13).

For $\mathcal L_k$, follow the integer quantity $\max(i,j)$ along the path.
It begins at zero, ends at $n-1$, and increases by at most one per step.
The first cell at which it equals $k$ has either $i=k,j\le k$ or
$j=k,i\le k$. The fixed-window restriction removes precisely the indices
below $\max(0,k-w)$, so that cell belongs to $\mathcal L_k$.

Path reversal proves the same statement for every $\mathcal R_k$. Every path
also visits every row index because its row coordinate starts at zero, ends
at $n-1$, and increases by at most one. Its visited cell in row $i$ obeys the
window, hence belongs to $\mathcal M_i$.

### Disjointness lemma

The selected sets are mutually disjoint.

- Every cell of $\mathcal L_k$ has maximum coordinate exactly $k$, so
  distinct left cuts cannot overlap.
- Every cell of $\mathcal R_k$ has minimum coordinate exactly $n-1-k$, so
  distinct right cuts cannot overlap.
- Left-cut coordinates are below $v$; right-cut coordinates are at least
  $n-v$. Condition (5) prevents overlap between those regions.
- A middle set has row coordinate in $[v,n-v)$, outside both cut regions.
  Distinct middle sets have distinct row coordinates.

The near-side restriction `j<k` in (11) is load-bearing. Extending an arm past
the diagonal can place one cell in adjacent cuts and invalidate the charging
argument.

## Enhanced admissibility

For a path cell $(i,j)$ write $C_{ij}=c_p(A_i,B_j)$. Define the directional
Enhanced bound by

$$
\begin{aligned}
E_{p,v}(A\mid B)
={}&
\sum_{k=0}^{v-1}\min_{(i,j)\in\mathcal L_k}C_{ij}
+
\sum_{k=0}^{v-1}\min_{(i,j)\in\mathcal R_k}C_{ij}\\
&+
\sum_{i=v}^{n-v-1}
d(A_i;L_i^B,U_i^B)^p.
\end{aligned}
\tag{14}
$$

For every path $P$, choose one crossed cell from every set. The crossing
lemma guarantees existence and the disjointness lemma guarantees that all
chosen cells are distinct. Each cut minimum is no greater than its chosen
cell cost. Equation (10) makes each middle projection no greater than the
cost of the chosen cell in that row. All other path costs are nonnegative, so

$$
E_{p,v}(A\mid B)
\le
\sum_{(i,j)\in P}C_{ij}.
\tag{15}
$$

This holds for every admissible $P$. Taking the minimum proves

$$
E_{p,v}(A\mid B)
\le
\operatorname{DTW}_{w,p}(A,B).
\tag{16}
$$

For $n=1$, production returns the forced-corner cost
$c_p(A_0,B_0)$, which equals DTW. Swapping the two series repeats the proof.
The symmetric maximum of two admissible directional values remains
admissible:

$$
E_{p,v}^{\mathrm{sym}}
=
\max(E_{p,v}(A\mid B),E_{p,v}(B\mid A))
\le
\operatorname{DTW}_{w,p}(A,B).
\tag{17}
$$

### Enhanced versus Keogh

At effective $v=1$, (14) replaces only the two endpoint Keogh projections by
the forced-corner costs. Because $B_0$ and $B_{n-1}$ belong to their endpoint
donor intervals, (10) gives

$$
E_{p,1}(A\mid B)
\ge
K_p(A\mid B),
\tag{18}
$$

where $K_p$ is directional LB_Keogh. The same direction holds after swapping
the series and therefore for the symmetric maxima.

For effective $v\ge2$, neither bound dominates. The exact L1 witnesses at
$w=1,V=2$ are:

| Series | Symmetric Enhanced | Symmetric Keogh |
|---|---:|---:|
| `A=(10,0,0,0)`, `B=(0,10,0,0)` | 10 | 0 |
| `A=(-1,-1,-1,-1)`, `B=(-1,-1,0,-1)` | 0 | 1 |

Both values are reproduced by independent set construction in the D3 oracle.
Consequently the live `Enhanced` cascade must evaluate both admissible bounds
and retain their maximum.

## Webb's four-point condition

The bridge/correction proof requires, for ordered values
$a\le x\le y\le b$,

$$
c_p(a,b)
\ge
c_p(a,y)+c_p(x,b)-c_p(x,y).
\tag{19}
$$

Write

$$
r_1=x-a,\qquad r_2=y-x,\qquad r_3=b-y,
\tag{20}
$$

with all three increments nonnegative. For L1, both sides of (19) are
$r_1+r_2+r_3$, so equality holds. For squared cost, the slack is

$$
\begin{aligned}
&(r_1+r_2+r_3)^2\\
&\quad-
\left((r_1+r_2)^2+(r_2+r_3)^2-r_2^2\right)
=2r_1r_3\ge0.
\end{aligned}
\tag{21}
$$

Thus both implemented costs satisfy the exact condition. No metric triangle
inequality is used.

## Exact all-index NoLR predicates and formula

The first pass is directional Keogh:

$$
K_p(A\mid B)
=
\sum_{i=0}^{n-1}d(A_i;L_i^B,U_i^B)^p.
\tag{22}
$$

Define the pointwise upper-safe predicate

$$
s_i^\uparrow=
\begin{cases}
\mathrm{false},&A_i>U_i^B,\\
[L_i^B\le LU_i^A],&A_i<L_i^B,\\
\mathrm{true},&L_i^B\le A_i\le U_i^B,
\end{cases}
\tag{23}
$$

and the lower-safe predicate

$$
s_i^\downarrow=
\begin{cases}
\mathrm{false},&A_i<L_i^B,\\
[U_i^B\ge UL_i^A],&A_i>U_i^B,\\
\mathrm{true},&L_i^B\le A_i\le U_i^B.
\end{cases}
\tag{24}
$$

Square brackets denote the truth value of the enclosed comparison. For
column $j$, the exact centered flags are

$$
F_j^\uparrow=\bigwedge_{i\in I_j}s_i^\uparrow,\qquad
F_j^\downarrow=\bigwedge_{i\in I_j}s_i^\downarrow.
\tag{25}
$$

The exact-predicate NoLR correction $q_j$ is the first applicable branch:

$$
q_j=
\begin{cases}
c_p(B_j,U_j^A),
&F_j^\uparrow\ \text{and}\ B_j>U_j^A,\\
c_p(B_j,L_j^A),
&F_j^\downarrow\ \text{and}\ B_j<L_j^A,\\
c_p(B_j,U_j^A)-c_p(UL_j^B,U_j^A),
&\neg F_j^\uparrow\ \text{and}\ B_j>UL_j^B\ge U_j^A,\\
c_p(B_j,L_j^A)-c_p(LU_j^B,L_j^A),
&\neg F_j^\downarrow\ \text{and}\ B_j<LU_j^B\le L_j^A,\\
0,&\text{otherwise}.
\end{cases}
\tag{26}
$$

The cases are evaluated in the displayed order. Production expresses the
two negations through `if`/`else if`: under the upper-overlap ordering,
the first full branch would already have fired if $F_j^\uparrow$ were true,
and the lower branch is impossible; the lower-overlap case is the reflection.
The paper writes strict $UL_j^B>U_j^A$ and $LU_j^B<L_j^A$ guards.
Production admits equality. This closure is conservative because the
subtracted cost is then zero and the same non-strict ordered-value proof
below applies.

The two full branches handle columns whose centered neighborhoods are safe
from conflicting bridge charges. When a neighborhood is not fully safe, the
upper or lower overlap branch subtracts the largest bridge overlap certified
by the appropriate secondary envelope. Every correction is nonnegative:
the full costs are nonnegative, and the ordering guards make the subtracted
cost no larger than the corresponding full cost.

## NoLR admissibility

Let $k_i$ denote row $i$'s summand in (22), and fix any admissible path
$P$. Every $(i,j)\in P$ satisfies both $i\in I_j$ and $j\in I_i$. We prove
the stronger cellwise statement

$$
k_i+q_j\le c_p(A_i,B_j)
\qquad\text{for every }(i,j)\in P.
\tag{27}
$$

Two elementary facts will be used repeatedly. First, (10) gives
$k_i\le c_p(A_i,B_j)$. Second, for $a\le x\le y\le b$,

$$
c_p(a,x)+c_p(y,b)\le c_p(a,b).
\tag{28}
$$

For L1 this follows by adding two disjoint subinterval lengths. For squared
cost, write the two lengths as $r,s\ge0$ with
$r+s\le b-a$; then $r^2+s^2\le(r+s)^2\le(b-a)^2$.

It remains to prove (27) for each nonzero branch of (26).

| Column branch | Possible position of $A_i$ | Ordered-value certificate |
|---|---|---|
| Full upper | Inside $[L_i^B,U_i^B]$ | $k_i=0$ and $A_i\le U_j^A<B_j$ |
| Full upper | Below $L_i^B$ | $A_i<L_i^B\le LU_i^A\le U_j^A<B_j$ |
| Full lower | Inside $[L_i^B,U_i^B]$ | $k_i=0$ and $B_j<L_j^A\le A_i$ |
| Full lower | Above $U_i^B$ | $B_j<L_j^A\le UL_i^A\le U_i^B<A_i$ |
| Upper overlap | Inside $[L_i^B,U_i^B]$ | $A_i\le U_j^A\le UL_j^B<B_j$ |
| Upper overlap | Below $L_i^B$ | $A_i\le U_j^A\le UL_j^B<B_j$ and $L_i^B\le UL_j^B$ |
| Lower overlap | Inside $[L_i^B,U_i^B]$ | $B_j<LU_j^B\le L_j^A\le A_i$ |
| Lower overlap | Above $U_i^B$ | $B_j<LU_j^B\le L_j^A\le A_i$ and $LU_j^B\le U_i^B$ |

The secondary-envelope comparisons in the table follow directly from
$j\in I_i$ or $i\in I_j$. For example,
$LU_i^A=\min_{r\in I_i}U_r^A\le U_j^A$ and
$UL_j^B=\max_{r\in I_j}L_r^B\ge L_i^B$.
The omitted full-upper/above and full-lower/below rows contradict their
respective conjunctions in (25). The omitted upper-overlap/above case would
require both $A_i>U_i^B\ge B_j$ and
$B_j>UL_j^B\ge U_j^A\ge A_i$; the lower-overlap/below case is its
reflection.

In a full branch, (28) applied to the corresponding ordered certificate
proves (27). In an overlap branch with $A_i$ inside its $B$ envelope,
$k_i=0$ and subtracting a nonnegative overlap only decreases the full
column cost, which is itself no greater than $c_p(A_i,B_j)$.

For the nontrivial upper-overlap/below row, monotonicity and the table give

$$
\begin{aligned}
k_i+q_j
&\le
c_p(A_i,UL_j^B)
+c_p(U_j^A,B_j)
-c_p(U_j^A,UL_j^B)\\
&\le c_p(A_i,B_j).
\end{aligned}
\tag{29}
$$

The last inequality is exactly (19) with
$A_i\le U_j^A\le UL_j^B\le B_j$. The lower-overlap/above row follows by
reflection, using
$B_j\le LU_j^B\le L_j^A\le A_i$. Thus (27) covers the cell types behind
Eqs. 26–43 of Webb and Petitjean, including the zero-correction case.

Every row and every column occurs at least once in a boundary-to-boundary
monotone path. Because $k_i,q_j\ge0$, summing (27) over all path cells gives

$$
\begin{aligned}
\sum_{(i,j)\in P}c_p(A_i,B_j)
&\ge
\sum_{(i,j)\in P}(k_i+q_j)\\
&\ge
\sum_{i=0}^{n-1}k_i+\sum_{j=0}^{n-1}q_j.
\end{aligned}
\tag{30}
$$

Since the argument holds for every admissible path, minimizing the left side
proves

$$
W_{p,\mathrm{NoLR}}(A\mid B)
:=
K_p(A\mid B)+\sum_{j=0}^{n-1}q_j
\le
\operatorname{DTW}_{w,p}(A,B).
\tag{31}
$$

This is the all-index NoLR proof. It does not insert `MinLRPaths` and does
not establish an order against full Algorithm 2.

Every $q_j\ge0$, so (22), (26), and (31) also give

$$
K_p(A\mid B)
\le
W_{p,\mathrm{NoLR}}(A\mid B)
\le
\operatorname{DTW}_{w,p}(A,B).
\tag{32}
$$

Taking the maximum after swapping $A$ and $B$ preserves admissibility and
dominates symmetric Keogh. A strict witness is
`A=(-1,-1)`, `B=(-1,2)`, $w=1$: the local result is 3 under
L1 and 9 under squared cost, while matching-direction Keogh is zero.

## The conservative trailing-flag cap

Production computes the exact pointwise predicates (23)–(24) as a trailing
run. For column $j$, it reads that run at

$$
t(j)=\min(j+w,n-1)
=j+\min(w,n-1-j).
\tag{33}
$$

The stored trailing interval is

$$
T_j=
\{\max(0,t(j)-2w),\ldots,t(j)\}.
\tag{34}
$$

If $j+w\le n-1$, then $t(j)=j+w$ and $T_j=I_j$. At the right tail,
$t(j)=n-1$. Since $j>n-1-w$,

$$
n-1-2w<j-w,
\tag{35}
$$

so the clipped trailing interval $T_j$ is a superset of the exact centered
interval $I_j$. A conjunction over the superset can be true only if the
conjunction over the subset is true:

$$
F_{j,\mathrm{cap}}^\uparrow
\Longrightarrow F_j^\uparrow,\qquad
F_{j,\mathrm{cap}}^\downarrow
\Longrightarrow F_j^\downarrow.
\tag{36}
$$

The cap therefore creates only false negatives. If an exact full flag becomes
false, production selects an overlap-subtracted correction or zero. In the
upper case,

$$
0
\le
c_p(B_j,U_j^A)-c_p(UL_j^B,U_j^A)
\le
c_p(B_j,U_j^A),
\tag{37}
$$

and the lower case is identical after reflection. Hence every capped
correction is no greater than its exact-predicate counterpart. Combining
(31), (36), and (37) proves (1).

The strict length-seven upper/lower witnesses give capped L1/squared values
20/400 and exact-predicate NoLR values 30/450. The supplementary binary
witness gives 3/3 versus 4/4.

### Relaxation gap and approximation statement

No modelling approximation is used in the Enhanced or exact NoLR proofs:
their exact-arithmetic leading-order modelling error is zero. The production
tail-cap is a discrete conservative relaxation, not a convergent numerical
approximation. Its exact gap is

$$
0
\le
W_{p,\mathrm{NoLR}}-W_{p,\mathrm{cap}}
\le
\sum_{j\in H}
\left(q_j^{\mathrm{full}}-q_j^{\mathrm{fallback}}\right),
\tag{38}
$$

where $H$ contains the at most $w$ tail columns whose exact true flag becomes
false. If all point costs in the finite fixture are bounded by $C$, then the
gap is at most $wC=O(wC)$. There is no universal relative or asymptotic error
bound, no small parameter, and no claim that this gap is negligible.

Ordinary floating accumulation has first-order absolute error on the scale
of $n\epsilon$ times the summed magnitudes. The overlap subtraction can
cancel, so no uniform relative-error bound follows. D17 retains compiler,
reassociation, and threshold decisions.

## Extreme-radius equivalence and F57

For any $w\ge n-1$, every donor window in (6) is global and every admissible
path cell is included. Equation (4) therefore preserves geometry exactly.
The CPU Webb implementation additionally uses an unsigned, saturating
doubled-radius threshold and free-run counters, and computes (30) by bounded
addition.

The primary F57 fixture
`A=(-2,-2)`, `B=(0,0)` has exact global L1/squared cost 4/8.
Radii $n-1$, $n$, and `INT_MAX` must agree. The nonconstant discriminator
`A=(0,0)`, `B=(-1,1)` has exact local L1/squared bound 2 at all three
radii. It catches an implementation that merely widens signed arithmetic
without normalizing the geometry.

The `2*n` scratch-size boundary, unrepresentable Enhanced length narrowing,
and mutable secondary-envelope shapes remain F46. Device-side signed window
arithmetic remains F50.

## Live cascade consequence

Both Enhanced and Keogh are admissible in the confirmed domain, so

$$
\max(K_p^{\mathrm{sym}},E_{p,v}^{\mathrm{sym}})
\le
\operatorname{DTW}_{w,p}.
\tag{39}
$$

The registered F54 fixture has three series ordered `{C,A,B}` at radius one.
The first two exact distances are zero. For the final pair, Kim and symmetric
Enhanced are zero, symmetric Keogh is 10, and DTW is 20. The repaired direct
and public routes both report three pairs, one envelope cutoff, one early
abandon, and two full evaluations while preserving the complete exact matrix.

This proves reachability and correctness, not an exact-matrix speedup. An
abandoned pair is recomputed without a cutoff because every matrix entry is
required.

## Complexity and storage

With precomputed primary envelopes, Enhanced evaluates its middle in $O(n)$
and its elastic cuts in

$$
O\!\left(\sum_{k=1}^{v-1}\min(k,w)\right)
=O(v\min(v,w)).
\tag{40}
$$

Its total evaluation cost is $O(n+v\min(v,w))$ and its additional storage is
$O(1)$. The usual fixed default $V=5$ makes evaluation linear in $n$.

Primary and secondary Webb envelopes are computed in $O(n)$ time and $O(n)$
storage. Both NoLR passes are linear. The reusable free-flag scratch contains
$2n$ bytes, so a bound evaluation after envelopes is $O(n)$ time and
$O(n)$ scratch.

## Executable oracle

The decisive oracle uses direct clipped-window scans, explicit Enhanced set
construction, direct safe predicates, and recursive monotone-path enumeration.
It does not call production DTW or duplicate its rolling recurrence.

| Subject | Exact inventory | Verdict |
|---|---:|---|
| Primary and secondary envelopes | 2,004 | zero mismatches |
| Explicit path, Webb, and tail cases | 35,982 | zero violations |
| Full-cover path cases | 7,380 | Delannoy checks included |
| Enhanced configurations | 68,787 | zero violations |
| Default `V=5` discriminators | 4/4 | exact |
| Webb correction branches | 4/4 | all reached |
| Webb strict witnesses | 2/2 | both metrics |
| Tail strict orientations | 2/2 | upper and lower |
| Four-point metric cases | 140 | L1 equality, squared slack |
| Enhanced/Keogh order witnesses | 2/2 | both directions |
| Direct/public cascade routes | 2/2 | exact matrix and counters |

The exact D3 marker is

```text
D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS
All tests passed (115 assertions in 1 test case)
```

The independent F57 target prints

```text
F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS
All tests passed (24 assertions in 1 test case)
```

Both targets clear CTest's skip-return policy, reject skip text, run serially,
and have finite timeouts. D3 also fixes `OMP_NUM_THREADS=1`.

## Code-conformance table

| Claim | Live artifact | Verdict |
|---|---|---|
| L1 and unrooted squared point costs, equation (3) | `dtwc/core/distance_metric.hpp` | **CONFIRMED** |
| Saturated Enhanced radius and band count, equations (4)–(5) | `dtwc/core/lower_bound_impl.hpp`, `lb_enhanced` | **CONFIRMED** for integer-representable lengths |
| Enhanced cut and middle formula, equations (11)–(14) | `dtwc/core/lower_bound_impl.hpp`, `lb_enhanced` | **CONFIRMED** by 68,787 exact configurations |
| Enhanced/Keogh maximum, equation (39) | `dtwc/core/pruned_distance_matrix.cpp` | **CONFIRMED** by both F54 routes |
| Primary and secondary envelopes, equations (7)–(8) | `compute_webb_envelope` | **CONFIRMED** by 2,004 direct scans |
| NoLR bridge and four corrections, equations (22)–(26) | `lb_webb` | **CONFIRMED** by 35,982 direct-predicate cases and 4/4 branches |
| Symmetric maxima, equations (17) and (32) | `lb_enhanced_symmetric`, `lb_webb_symmetric` | **CONFIRMED** |
| Conservative tail lookup, equations (33)–(38) | `lb_webb` bounded `idx` and free flags | **CONFIRMED** with two strict orientations |
| Extreme-radius arithmetic | `lb_webb` effective `w`, `two_w`, saturated counters | **CONFIRMED** by F57 focused target |
| Non-skippable execution contracts | `tests/CMakeLists.txt` | **CONFIRMED** for D3 and F57 |
| Public envelope shape, source, radius, and alias validation | `Envelope`, `WebbEnvelope`, span wrappers | **DISCREPANCY** F46 |
| GPU extreme-radius envelope arithmetic | CUDA and Metal envelope kernels | **DISCREPANCY/OPEN** F50 |
| Floating overlap subtraction and cutoff equality | lower bounds and pruned matrix route | **OPEN** D17 |

## Scope verdicts

- **CONFIRMED:** Enhanced admissibility for (2)–(5), directional/symmetric
  forms, both implemented point costs, and effective `V=1` Keogh dominance.
- **CONFIRMED:** no universal Enhanced/Keogh order for effective `V>=2`, by
  two exact strict witnesses.
- **CONFIRMED:** local exact-predicate NoLR admissibility, directional Keogh
  dominance, and symmetric Keogh dominance under the four-point condition for
  L1 and unrooted squared L2.
- **CONFIRMED:** the production trailing cap is no greater than
  exact-predicate NoLR and remains admissible.
- **CONFIRMED:** F54's live take-maximum route and F57's focused Windows CPU
  global-radius behavior.
- **NOT CLAIMED:** any order between local NoLR and full Algorithm 2, custom
  point costs, unequal or multivariate series, nonfinite values, other DTW
  variants, public provenance safety, GPU extreme-radius behavior, or
  bit-identical floating threshold decisions.

The preregistered bands, red-first results, two product attempts, exact
terminal output, and remaining integration requirements are preserved in
`.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
