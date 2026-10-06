// Per-pair kernels, dtwc_cl (O3, -march=native, ThinLTO). Kernel 1 = dtw_kernel_linear (dtw_kernel.hpp:261-268),
// kernel 2 = dtw_kernel_banded (dtw_kernel.hpp:333-340). Chain: left -> fcmp -> fcsel -> fadd -> left.

// ---- linear f64 L1 (dtwc_cl)
LOOP 1000ad708-1000ad738: 13 insns, calls/traps []
    ldr       d4, [x13]
    ldr       d5, [x14], #0x8
    ldr       d6, [x21, x12, lsl #3]
    fabd      d5, d5, d6
    fcmp      d4, d3
    fcsel     d3, d4, d3, mi
    fcmp      d2, d3
    fcsel     d2, d2, d3, mi
    fadd      d2, d2, d5
    str       d2, [x13], #0x8
    mov.16b   v3, v4
    subs      x15, x15, #0x1
    b.ne      0x1000ad708


// ---- banded f64 L1 (dtwc_cl)
LOOP 1000ad554-1000ad584: 13 insns, calls/traps []
    ldr       d3, [x17]
    ldr       d4, [x9, x13, lsl #3]
    ldr       d5, [x16], #0x8
    fabd      d4, d4, d5
    fcmp      d3, d2
    fcsel     d2, d3, d2, mi
    fcmp      d1, d2
    fcsel     d1, d1, d2, mi
    fadd      d1, d1, d4
    str       d1, [x17], #0x8
    mov.16b   v2, v3
    subs      x15, x15, #0x1
    b.ne      0x1000ad554


// ---- linear f64 squared (dtwc_cl)
LOOP 1000ad2a0-1000ad2d4: 14 insns, calls/traps []
    ldr       d4, [x14], #0x8
    ldr       d5, [x21, x12, lsl #3]
    ldr       d6, [x13]
    fsub      d4, d4, d5
    fmul      d4, d4, d4
    fcmp      d6, d2
    fcsel     d2, d6, d2, mi
    fcmp      d3, d2
    fcsel     d2, d3, d2, mi
    fadd      d3, d2, d4
    str       d3, [x13], #0x8
    mov.16b   v2, v6
    subs      x15, x15, #0x1
    b.ne      0x1000ad2a0


// ---- linear f32 L1 (dtwc_cl)
LOOP 1000bb25c-1000bb28c: 13 insns, calls/traps []
    ldr       s4, [x13]
    ldr       s5, [x14], #0x4
    ldr       s6, [x21, x12, lsl #2]
    fabd      s5, s5, s6
    fcmp      s4, s3
    fcsel     s3, s4, s3, mi
    fcmp      s2, s3
    fcsel     s2, s2, s3, mi
    fadd      s2, s2, s5
    str       s2, [x13], #0x4
    mov.16b   v3, v4
    subs      x15, x15, #0x1
    b.ne      0x1000bb25c


// ---- linear f64 ADTW (dtwc_cl, separate symbol; chain fadd+fcmp+fcsel+fadd)
LOOP 1000b78dc-1000b7908: 12 insns, calls/traps []
    ldr       d3, [x19, x9, lsl #3]
    ldr       d4, [x21]
    fabd      d3, d3, d4
    fadd      d2, d2, d8
    fminnm    d2, d2, d1
    fcmp      d0, d2
    fcsel     d2, d0, d2, mi
    fadd      d2, d2, d3
    str       d2, [x8, x9, lsl #3]
    add       x9, x9, #0x1
    cmp       x20, x9
    b.ne      0x1000b78dc
LOOP 1000b7968-1000b79a0: 15 insns, calls/traps []
    ldr       d4, [x13]
    ldr       d5, [x14], #0x8
    ldr       d6, [x21, x12, lsl #3]
    fabd      d5, d5, d6
    fadd      d6, d4, d8
    fadd      d3, d3, d8
    fcmp      d6, d2
    fcsel     d2, d6, d2, mi
    fcmp      d3, d2
    fcsel     d2, d3, d2, mi
    fadd      d3, d2, d5
    str       d3, [x13], #0x8
    mov.16b   v2, v4
    subs      x15, x15, #0x1
    b.ne      0x1000b7968
LOOP 1000b79d4-1000b79f8: 10 insns, calls/traps []
    ldr       d2, [x19]
    ldr       d3, [x10], #0x8
    fabd      d2, d2, d3
    fadd      d0, d0, d8
    fcmp      d0, d1
    fcsel     d0, d0, d1, mi
    fadd      d0, d2, d0
    str       d0, [x8]
    subs      x9, x9, #0x1
    b.ne      0x1000b79d4


// ---- wheel f64 L1 Standard per-pair loop (_dtwcpp_core.so @1ec44, also @1ee58): the -Os copy, early-abandon row minimum not unswitched
LOOP 1ec44-1ec80: 16 insns, calls/traps []
    ldr       d6, [x14]
    ldr       d7, [x20, x11, lsl #3]
    ldr       d16, [x13], #0x8
    fabd      d7, d7, d16
    fcmp      d6, d5
    fcsel     d5, d6, d5, mi
    fcmp      d4, d5
    fcsel     d4, d4, d5, mi
    fadd      d4, d4, d7
    fcmp      d4, d3
    fccmp     d8, d1, #0x8, mi
    str       d4, [x14], #0x8
    fcsel     d3, d4, d3, ge
    mov.16b   v5, v6
    subs      x12, x12, #0x1
    b.ne      0x1ec44

// ---- same source relinked without -Os in the binding TU (scratch nominsize/_dtwcpp_core...so @26734)
LOOP 26734-26764: 13 insns, calls/traps []
    ldr       d3, [x0]
    ldr       d4, [x9, x15, lsl #3]
    ldr       d5, [x17], #0x8
    fabd      d4, d4, d5
    fcmp      d3, d2
    fcsel     d2, d3, d2, mi
    fcmp      d1, d2
    fcsel     d1, d1, d2, mi
    fadd      d1, d1, d4
    str       d1, [x0], #0x8
    mov.16b   v2, v3
    subs      x16, x16, #0x1
    b.ne      0x26734

