# 2026-09-29 — Y2: one DistanceMatrix, the lookup and the llfio include count

Question: after DenseDistanceMatrix and MmapDistanceMatrix became one `core::DistanceMatrix`
(raw `double *`, heap or mapped), is a matrix lookup index arithmetic and one load, with no
branch on the storage kind and no call; and how many translation units still parse llfio?

## Lookup assembly [confirmed]

Probe TU: `probe_get(m, i, j)` returns one element; before it read `Problem::distMat_t` through
`std::visit` as `Problem::visit_distmat` did, after it calls `DistanceMatrix::get`. Compiled with
the flags of `dtwc/Problem.cpp` from `build/compile_commands.json` minus `-flto=thin`, plus
`-S -masm=intel` (clang 21, Windows, `-O3 -march=native`). Probe sources and full `.s` files
were in the session scratchpad; the functions are reproduced here verbatim (SEH directives dropped).

Before (base 4dcc39d):

```asm
	sub	rsp, 40
	movzx	eax, byte ptr [rcx + 296]      ; variant index
	test	eax, eax
	je	.LBB80_3
	cmp	eax, 1
	jne	.LBB80_4
	add	rcx, 240                        ; the Mmap alternative
.LBB80_3:
	cmp	rdx, r8
	mov	rax, r8
	cmovb	rax, rdx
	cmova	r8, rdx
	lea	rdx, [r8 + 1]
	imul	rdx, r8
	and	rdx, -2
	shl	rdx, 2
	add	rdx, qword ptr [rcx]
	vmovsd	xmm0, qword ptr [rdx + 8*rax]
	add	rsp, 40
	ret
.LBB80_4:
	call	"??$_Dispatch2@..."            ; valueless variant: bad_variant_access
	int3
```

After:

```asm
	cmp	rdx, r8
	mov	rax, r8
	cmovb	rax, rdx
	cmova	r8, rdx
	lea	rdx, [r8 + 1]
	imul	rdx, r8
	and	rdx, -2
	shl	rdx, 2
	add	rdx, qword ptr [rcx + 32]      ; data_
	vmovsd	xmm0, qword ptr [rdx + 8*rax]
	ret
```

The i<j swap is two `cmov`s, not a branch. A row loop (`s += m.get(i, j)`, j = 0..n) vectorises to
the same `vgatherqpd` body before and after; before, clang unswitched it into one copy per variant
alternative behind the index test. The probe TU's assembly was 25,217 lines before (llfio's inline
functions, through `Problem.hpp`) and 754 after.

## Translation units that include llfio [confirmed]

`ninja -C build -t deps`, objects whose dependency list names an llfio header:

| | objects | include llfio |
| --- | --- | --- |
| base 4dcc39d (llfio superbuild) | 517 | 130 (16 dtwc++, 6 mip-solvers, the rest tests and tools) |
| after | 518 | 1 (`dtwc++/core/distance_matrix.cpp`) |
