// Scratch variants (probes/kbench/v_*.hpp), compiled with the dtw_lanes.cpp command (kbench_native).

// ---- lanes f64 L1, v_fmin (std::fmin -> fminnm.2d), W=8: 28 insns; chain left -> fminnm -> fadd (5 cycles)
LOOP 100006bc4-100006c30: 28 insns, calls/traps []
    ldur      q21, [x2, #-0x20]
    ld1r.2d   { v22 }, [x1], #8
    fabd.2d   v23, v22, v1
    fminnm.2d v6, v6, v21
    fminnm.2d v5, v6, v5
    fadd.2d   v5, v5, v23
    ldp       q23, q24, [x2, #-0x10]
    fminnm.2d v6, v16, v23
    fminnm.2d v6, v6, v7
    fabd.2d   v7, v22, v2
    fadd.2d   v7, v6, v7
    stp       q5, q7, [x2, #-0x20]
    fabd.2d   v6, v22, v3
    fminnm.2d v16, v17, v24
    fminnm.2d v16, v16, v18
    fadd.2d   v18, v16, v6
    ldr       q25, [x2, #0x10]
    fabd.2d   v6, v22, v4
    fminnm.2d v16, v19, v25
    fminnm.2d v16, v16, v20
    fadd.2d   v20, v16, v6
    stp       q18, q20, [x2], #0x40
    mov.16b   v6, v21
    mov.16b   v16, v23
    mov.16b   v17, v24
    mov.16b   v19, v25
    subs      x0, x0, #0x1
    b.ne      0x100006bc4

// ---- lanes f64 L1, v_w16 (W=16): 72 insns, one stack reload per row (ldr q30, [sp, #0x30])
LOOP 10000cc78-10000cd94: 72 insns, calls/traps []
    ldur      q9, [x2, #-0x40]
    ld1r.2d   { v12 }, [x1], #8
    fabd.2d   v10, v12, v30
    fcmgt.2d  v11, v21, v9
    bit.16b   v21, v9, v11
    fcmgt.2d  v11, v21, v17
    bif.16b   v17, v21, v11
    fadd.2d   v17, v17, v10
    ldp       q10, q11, [x2, #-0x30]
    fcmgt.2d  v21, v22, v10
    bsl.16b   v21, v10, v22
    fcmgt.2d  v22, v21, v18
    bif.16b   v18, v21, v22
    fabd.2d   v21, v12, v13
    fadd.2d   v18, v18, v21
    stp       q17, q18, [x2, #-0x40]
    fabd.2d   v21, v12, v14
    fcmgt.2d  v22, v23, v11
    bsl.16b   v22, v11, v23
    fcmgt.2d  v23, v22, v19
    bif.16b   v19, v22, v23
    fadd.2d   v19, v19, v21
    ldp       q13, q14, [x2, #-0x10]
    fcmgt.2d  v21, v24, v13
    bsl.16b   v21, v13, v24
    fcmgt.2d  v22, v21, v20
    bif.16b   v20, v21, v22
    fabd.2d   v21, v12, v4
    fadd.2d   v20, v20, v21
    stp       q19, q20, [x2, #-0x20]
    fabd.2d   v21, v12, v5
    fcmgt.2d  v22, v27, v14
    bsl.16b   v22, v14, v27
    fcmgt.2d  v23, v22, v25
    bit.16b   v22, v25, v23
    fadd.2d   v25, v22, v21
    ldp       q15, q30, [x2, #0x10]
    fcmgt.2d  v21, v28, v15
    bsl.16b   v21, v15, v28
    fcmgt.2d  v22, v21, v26
    bit.16b   v21, v26, v22
    fabd.2d   v22, v12, v6
    fadd.2d   v26, v21, v22
    stp       q25, q26, [x2]
    fabd.2d   v21, v12, v7
    fcmgt.2d  v22, v0, v30
    bit.16b   v0, v30, v22
    fcmgt.2d  v22, v0, v29
    bit.16b   v0, v29, v22
    fadd.2d   v29, v0, v21
    ldr       q8, [x2, #0x30]
    fabd.2d   v0, v12, v16
    fcmgt.2d  v21, v1, v8
    bit.16b   v1, v8, v21
    fcmgt.2d  v21, v1, v31
    bit.16b   v1, v31, v21
    fadd.2d   v31, v1, v0
    stp       q29, q31, [x2, #0x20]
    add       x2, x2, #0x80
    mov.16b   v21, v9
    mov.16b   v22, v10
    mov.16b   v23, v11
    mov.16b   v24, v13
    mov.16b   v13, v3
    mov.16b   v27, v14
    mov.16b   v14, v2
    mov.16b   v28, v15
    mov.16b   v0, v30
    ldr       q30, [sp, #0x30]
    mov.16b   v1, v8
    subs      x0, x0, #0x1
    b.ne      0x10000cc78

// ---- lanes f64 L1, v_fmin_w16: 56 insns = 8 fabd + 16 fminnm + 8 fadd (32 FP ops per 16 cells), one stack reload
LOOP 10001b18c-10001b268: 56 insns, calls/traps []
    ldur      q9, [x2, #-0x40]
    ld1r.2d   { v12 }, [x1], #8
    fabd.2d   v10, v12, v30
    fminnm.2d v19, v19, v9
    fminnm.2d v17, v19, v17
    fadd.2d   v17, v17, v10
    ldp       q10, q11, [x2, #-0x30]
    fminnm.2d v19, v21, v10
    fminnm.2d v18, v19, v18
    fabd.2d   v19, v12, v13
    fadd.2d   v18, v18, v19
    stp       q17, q18, [x2, #-0x40]
    fabd.2d   v19, v12, v14
    fminnm.2d v21, v23, v11
    fminnm.2d v20, v21, v20
    fadd.2d   v20, v20, v19
    ldp       q13, q14, [x2, #-0x10]
    fminnm.2d v19, v24, v13
    fminnm.2d v19, v19, v22
    fabd.2d   v21, v12, v4
    fadd.2d   v22, v19, v21
    stp       q20, q22, [x2, #-0x20]
    fabd.2d   v19, v12, v5
    fminnm.2d v21, v26, v14
    fminnm.2d v21, v21, v25
    fadd.2d   v25, v21, v19
    ldp       q15, q30, [x2, #0x10]
    fminnm.2d v19, v28, v15
    fminnm.2d v19, v19, v27
    fabd.2d   v21, v12, v6
    fadd.2d   v27, v19, v21
    stp       q25, q27, [x2]
    fabd.2d   v19, v12, v7
    fminnm.2d v0, v0, v30
    fminnm.2d v0, v0, v29
    fadd.2d   v29, v0, v19
    ldr       q8, [x2, #0x30]
    fabd.2d   v0, v12, v16
    fminnm.2d v1, v1, v8
    fminnm.2d v1, v1, v31
    fadd.2d   v31, v1, v0
    stp       q29, q31, [x2, #0x20]
    add       x2, x2, #0x80
    mov.16b   v19, v9
    mov.16b   v21, v10
    mov.16b   v23, v11
    mov.16b   v24, v13
    mov.16b   v13, v3
    mov.16b   v26, v14
    mov.16b   v14, v2
    mov.16b   v28, v15
    mov.16b   v0, v30
    ldr       q30, [sp, #0x30]
    mov.16b   v1, v8
    subs      x0, x0, #0x1
    b.ne      0x10001b18c

// ---- lanes f64 L1, v_w32 (W=32): 26 stack loads/stores per row (register spills)
LOOP 100015270-1000154e8: 159 insns, calls/traps []
    str       q29, [sp, #0x120]
    mov.16b   v0, v1
    mov.16b   v1, v2
    ld1r.2d   { v26 }, [x7], #8
    ldp       q2, q30, [x21, #-0x80]
    stp       q30, q2, [sp, #0x140]
    fcmgt.2d  v29, v0, v2
    bit.16b   v0, v2, v29
    fcmgt.2d  v29, v0, v25
    bif.16b   v25, v0, v29
    fcmgt.2d  v0, v1, v30
    bsl.16b   v0, v30, v1
    fcmgt.2d  v1, v0, v24
    bit.16b   v0, v24, v1
    str       q0, [sp, #0x130]
    mov.16b   v0, v31
    mov.16b   v1, v8
    mov.16b   v24, v3
    mov.16b   v3, v11
    ldp       q31, q8, [x21, #-0x60]
    fcmgt.2d  v29, v0, v31
    bit.16b   v0, v31, v29
    fcmgt.2d  v29, v0, v23
    bif.16b   v23, v0, v29
    fcmgt.2d  v0, v1, v8
    bsl.16b   v0, v8, v1
    fcmgt.2d  v1, v0, v22
    bif.16b   v22, v0, v1
    ldr       q0, [sp, #0x160]
    ldur      q30, [x21, #-0x40]
    str       q30, [sp, #0x160]
    mov.16b   v2, v10
    ldur      q10, [x21, #-0x30]
    fcmgt.2d  v29, v0, v30
    bit.16b   v0, v30, v29
    fcmgt.2d  v29, v0, v21
    bif.16b   v21, v0, v29
    fcmgt.2d  v0, v9, v10
    bsl.16b   v0, v10, v9
    fcmgt.2d  v1, v0, v20
    bif.16b   v20, v0, v1
    ldp       q1, q0, [sp, #0x170]
    ldp       q30, q9, [x21, #-0x20]
    fcmgt.2d  v29, v0, v30
    stp       q9, q30, [sp, #0x170]
    bit.16b   v0, v30, v29
    fcmgt.2d  v29, v0, v18
    bif.16b   v18, v0, v29
    fcmgt.2d  v0, v1, v9
    bsl.16b   v0, v9, v1
    mov.16b   v9, v10
    fcmgt.2d  v1, v0, v16
    bif.16b   v16, v0, v1
    mov.16b   v0, v15
    ldp       q15, q11, [x21]
    fcmgt.2d  v29, v0, v15
    bit.16b   v0, v15, v29
    fcmgt.2d  v29, v0, v19
    bif.16b   v19, v0, v29
    fcmgt.2d  v0, v3, v11
    bsl.16b   v0, v11, v3
    fcmgt.2d  v1, v0, v17
    bif.16b   v17, v0, v1
    mov.16b   v0, v14
    ldp       q14, q10, [x21, #0x20]
    fcmgt.2d  v29, v0, v14
    bit.16b   v0, v14, v29
    fcmgt.2d  v29, v0, v7
    bif.16b   v7, v0, v29
    fcmgt.2d  v0, v2, v10
    bsl.16b   v0, v10, v2
    fcmgt.2d  v1, v0, v6
    bif.16b   v6, v0, v1
    mov.16b   v0, v13
    mov.16b   v1, v28
    ldp       q13, q28, [x21, #0x40]
    fcmgt.2d  v29, v0, v13
    bit.16b   v0, v13, v29
    fcmgt.2d  v29, v0, v5
    bit.16b   v0, v5, v29
    fcmgt.2d  v5, v1, v28
    bit.16b   v1, v28, v5
    fcmgt.2d  v5, v1, v4
    bit.16b   v1, v4, v5
    mov.16b   v4, v12
    mov.16b   v5, v27
    ldp       q12, q27, [x21, #0x60]
    fcmgt.2d  v29, v4, v12
    bit.16b   v4, v12, v29
    ldr       q2, [sp, #0x120]
    fcmgt.2d  v29, v4, v2
    mov.16b   v3, v29
    bsl.16b   v3, v2, v4
    fcmgt.2d  v4, v5, v27
    bsl.16b   v4, v27, v5
    fcmgt.2d  v5, v4, v24
    mov.16b   v2, v5
    bsl.16b   v2, v24, v4
    ldr       q4, [sp, #0x110]
    fabd.2d   v4, v26, v4
    fadd.2d   v25, v25, v4
    ldr       q4, [sp, #0x100]
    fabd.2d   v4, v26, v4
    ldr       q5, [sp, #0x130]
    fadd.2d   v24, v5, v4
    ldr       q4, [sp, #0xf0]
    fabd.2d   v4, v26, v4
    fadd.2d   v23, v23, v4
    ldr       q4, [sp, #0xe0]
    fabd.2d   v4, v26, v4
    fadd.2d   v22, v22, v4
    ldr       q4, [sp, #0xd0]
    fabd.2d   v4, v26, v4
    fadd.2d   v21, v21, v4
    ldr       q4, [sp, #0xc0]
    fabd.2d   v4, v26, v4
    fadd.2d   v20, v20, v4
    ldr       q4, [sp, #0xb0]
    fabd.2d   v4, v26, v4
    fadd.2d   v18, v18, v4
    ldr       q4, [sp, #0xa0]
    fabd.2d   v4, v26, v4
    fadd.2d   v16, v16, v4
    ldr       q4, [sp, #0x90]
    fabd.2d   v4, v26, v4
    fadd.2d   v19, v19, v4
    ldr       q4, [sp, #0x80]
    fabd.2d   v4, v26, v4
    fadd.2d   v17, v17, v4
    ldr       q4, [sp, #0x70]
    fabd.2d   v4, v26, v4
    fadd.2d   v7, v7, v4
    ldr       q4, [sp, #0x60]
    fabd.2d   v4, v26, v4
    fadd.2d   v6, v6, v4
    ldr       q4, [sp, #0x50]
    fabd.2d   v4, v26, v4
    fadd.2d   v5, v0, v4
    ldr       q0, [sp, #0x40]
    fabd.2d   v0, v26, v0
    fadd.2d   v4, v1, v0
    ldr       q0, [sp, #0x30]
    fabd.2d   v0, v26, v0
    fadd.2d   v29, v3, v0
    ldr       q0, [sp, #0x20]
    fabd.2d   v0, v26, v0
    fadd.2d   v3, v2, v0
    ldp       q2, q1, [sp, #0x140]
    stp       q25, q24, [x21, #-0x80]
    stp       q23, q22, [x21, #-0x60]
    stp       q21, q20, [x21, #-0x40]
    stp       q18, q16, [x21, #-0x20]
    stp       q19, q17, [x21]
    stp       q7, q6, [x21, #0x20]
    stp       q5, q4, [x21, #0x40]
    stp       q29, q3, [x21, #0x60]
    add       x21, x21, #0x100
    subs      x6, x6, #0x1
    b.ne      0x100015270


// ---- per-pair linear f64 L1, v_skew2: two columns per pass, 21 insns per 2 cells, two chains
LOOP 1000246b8-100024708: 21 insns, calls/traps []
    ldr       d5, [x9]
    ldr       d6, [x15], #0x8
    ldr       d7, [x21, x13, lsl #3]
    fabd      d7, d6, d7
    fcmp      d5, d4
    fcsel     d4, d5, d4, mi
    fcmp      d3, d4
    fcsel     d4, d3, d4, mi
    ldr       d16, [x21, x14, lsl #3]
    fadd      d4, d4, d7
    fabd      d6, d6, d16
    fcmp      d4, d3
    fcsel     d3, d4, d3, mi
    fcmp      d2, d3
    fcsel     d2, d2, d3, mi
    fadd      d2, d2, d6
    str       d2, [x9], #0x8
    mov.16b   v3, v4
    mov.16b   v4, v5
    subs      x16, x16, #0x1
    b.ne      0x1000246b8

// ---- per-pair linear f64 L1, v_fmin: 11 insns per cell
LOOP 1000233c4-1000233ec: 11 insns, calls/traps []
    ldr       d4, [x13]
    ldr       d5, [x14], #0x8
    ldr       d6, [x21, x12, lsl #3]
    fabd      d5, d5, d6
    fminnm    d2, d2, d4
    fminnm    d2, d2, d3
    fadd      d3, d5, d2
    str       d3, [x13], #0x8
    mov.16b   v2, v4
    subs      x15, x15, #0x1
    b.ne      0x1000233c4

