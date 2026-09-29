# run_openmp: per-thread failure slots, inner loop before and after [confirmed]

Question: does replacing run_openmp's lowest-index capture (atomic CAS cutoff + unnamed `omp critical`)
with one failure slot per thread add per-iteration work to the OpenMP loop that every parallel fill runs?
Criterion, set before compiling: the inner loop gains no call, no load of shared state, no spill or
reload, and no branch beyond the "my slot is set" test.

Code: base `ff53782` vs pb/Y3 (W6c). Compiler: clang 21.1.8, the flags `build/compile_commands.json`
records for `dtwc/Problem.cpp` (clang-win preset, Release), without `-flto=thin`, plus
`-S -mllvm --x86-asm-syntax=intel` (`-masm=intel` also switches the inline-asm dialect, which breaks
llfio's AT&T inline asm). Loop: the outlined region of `run_openmp<fill_row>` in
`Problem::fillDistanceMatrix_BruteForce`; `fill_row` is not inlined, so the loop is the scheduling
overhead around one call per row.

```text
== before (i spilled; the shared atomic cutoff loaded with acquire each row)
.LBB460_8:  movsxd rax, dword ptr [rbp - 8]          ; ub
            mov    rcx, qword ptr [rbp - 40]         ; reload i
            cmp rcx, rax ; lea rcx, [rcx + 1] ; jge .LBB460_9
.LBB460_6:  mov    qword ptr [rbp - 40], rcx         ; spill i
            mov    rax, qword ptr [rbp - 32]         ; reload &earliest_failure
            movsxd rax, dword ptr [rax]              ; load shared atomic (#MEMBARRIER)
            cmp rcx, rax ; jg .LBB460_8
            mov    rcx, qword ptr [rbp + 80]
            mov    rdx, qword ptr [rbp - 40]         ; reload i
            call   fill_row ; jmp .LBB460_8
== after (same shape; the thread's own slot flag replaces the shared atomic)
.LBB460_8:  mov    rax, qword ptr [rbp - 16]         ; reload i
            cmp    rax, qword ptr [rbp - 8]          ; ub
            lea rax, [rax + 1] ; mov qword ptr [rbp - 16], rax (spill i) ; jge .LBB460_9
.LBB460_6:  mov    rax, qword ptr [rbp - 32]         ; reload &slot
            cmp    byte ptr [rax + 16], 0            ; slot.failed
            jne .LBB460_8
            mov    rcx, qword ptr [rbp + 104]
            mov    rdx, qword ptr [rbp - 16]         ; reload i
            call   fill_row ; jmp .LBB460_8
```

Result: per row, before and after both do one call, one spill and three reloads, two compare-and-branch
pairs and one ub load; the acquire load of the shared cutoff becomes a plain byte load of the thread's
own slot. The loop counter is now 64-bit (`__kmpc_dispatch_init_8` / `next_8`). A probe TU with a task the
compiler inlines (`out[i] = i * 0.5`) shows the same: the skip test is one `cmp byte ptr [rbp], 0` where
the base had the atomic load, and nothing else changes.

Rejected on the way: a thread-local `bool failed` instead of the slot flag. Under Windows EH a local
written in the catch funclet and read after the invoke lives in the frame, so clang added two spills
and a reload per row (a speculative `failed = 1` store before each call, `failed = 0` after it). Kept in
memory behind the slot reference, the flag costs the one load the old cutoff cost. The flag exists at
all because the MSVC STL's `exception_ptr::operator bool` is an out-of-line call
(`__ExceptionPtrToBool`), so testing the exception_ptr itself would add a call per row on Windows.
