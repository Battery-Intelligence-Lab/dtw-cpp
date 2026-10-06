// Head pb/arm-lanes: the lanes inner loops of the linked build/bin/dtwc_cl (ThinLTO, -march=native), one DP row per iteration.
// objdump -d --no-show-raw-insn bin/dtwc_cl | lane_loops.py <listing> ld1r (the 2026-10-06 audit's probe)

// ---- lanes f64 squared, W=16: 64 insns per row (4.00 per cell), 16 min instructions, 1 stack access(es)
LOOP 1000ec1a0-1000ec29c: 64 insns, calls/traps []
    ldur      q9, [x1, #-0x40]
    ld1r.2d   { v12 }, [x0], #8
    fsub.2d   v10, v12, v30
    fmul.2d   v10, v10, v10
    fminnm.2d v19, v19, v9
    fminnm.2d v17, v19, v17
    fadd.2d   v17, v17, v10
    ldp       q10, q11, [x1, #-0x30]
    fminnm.2d v19, v21, v10
    fminnm.2d v18, v19, v18
    fsub.2d   v19, v12, v13
    fmul.2d   v19, v19, v19
    fadd.2d   v18, v18, v19
    stp       q17, q18, [x1, #-0x40]
    fsub.2d   v19, v12, v14
    fmul.2d   v19, v19, v19
    fminnm.2d v21, v23, v11
    fminnm.2d v20, v21, v20
    fadd.2d   v20, v20, v19
    ldp       q13, q14, [x1, #-0x10]
    fminnm.2d v19, v24, v13
    fminnm.2d v19, v19, v22
    fsub.2d   v21, v12, v4
    fmul.2d   v21, v21, v21
    fadd.2d   v22, v19, v21
    stp       q20, q22, [x1, #-0x20]
    fsub.2d   v19, v12, v5
    fmul.2d   v19, v19, v19
    fminnm.2d v21, v26, v14
    fminnm.2d v21, v21, v25
    fadd.2d   v25, v21, v19
    ldp       q15, q30, [x1, #0x10]
    fminnm.2d v19, v28, v15
    fminnm.2d v19, v19, v27
    fsub.2d   v21, v12, v6
    fmul.2d   v21, v21, v21
    fadd.2d   v27, v19, v21
    stp       q25, q27, [x1]
    fsub.2d   v19, v12, v7
    fmul.2d   v19, v19, v19
    fminnm.2d v0, v0, v30
    fminnm.2d v0, v0, v29
    fadd.2d   v29, v0, v19
    ldr       q8, [x1, #0x30]
    fsub.2d   v0, v12, v16
    fmul.2d   v0, v0, v0
    fminnm.2d v1, v1, v8
    fminnm.2d v1, v1, v31
    fadd.2d   v31, v1, v0
    stp       q29, q31, [x1, #0x20]
    add       x1, x1, #0x80
    mov.16b   v19, v9
    mov.16b   v21, v10
    mov.16b   v23, v11
    mov.16b   v24, v13
    mov.16b   v13, v3
    mov.16b   v26, v14
    mov.16b   v14, v2
    mov.16b   v28, v15
    mov.16b   v0, v30
    ldr       q30, [sp, #0x80]
    mov.16b   v1, v8
    subs      x17, x17, #0x1
    b.ne      0x1000ec1a0

// ---- lanes f64 L1, W=16: 56 insns per row (3.50 per cell), 16 min instructions, 1 stack access(es)
LOOP 1000ed910-1000ed9ec: 56 insns, calls/traps []
    ldur      q9, [x1, #-0x40]
    ld1r.2d   { v12 }, [x0], #8
    fabd.2d   v10, v12, v30
    fminnm.2d v19, v19, v9
    fminnm.2d v17, v19, v17
    fadd.2d   v17, v17, v10
    ldp       q10, q11, [x1, #-0x30]
    fminnm.2d v19, v21, v10
    fminnm.2d v18, v19, v18
    fabd.2d   v19, v12, v13
    fadd.2d   v18, v18, v19
    stp       q17, q18, [x1, #-0x40]
    fabd.2d   v19, v12, v14
    fminnm.2d v21, v23, v11
    fminnm.2d v20, v21, v20
    fadd.2d   v20, v20, v19
    ldp       q13, q14, [x1, #-0x10]
    fminnm.2d v19, v24, v13
    fminnm.2d v19, v19, v22
    fabd.2d   v21, v12, v4
    fadd.2d   v22, v19, v21
    stp       q20, q22, [x1, #-0x20]
    fabd.2d   v19, v12, v5
    fminnm.2d v21, v26, v14
    fminnm.2d v21, v21, v25
    fadd.2d   v25, v21, v19
    ldp       q15, q30, [x1, #0x10]
    fminnm.2d v19, v28, v15
    fminnm.2d v19, v19, v27
    fabd.2d   v21, v12, v6
    fadd.2d   v27, v19, v21
    stp       q25, q27, [x1]
    fabd.2d   v19, v12, v7
    fminnm.2d v0, v0, v30
    fminnm.2d v0, v0, v29
    fadd.2d   v29, v0, v19
    ldr       q8, [x1, #0x30]
    fabd.2d   v0, v12, v16
    fminnm.2d v1, v1, v8
    fminnm.2d v1, v1, v31
    fadd.2d   v31, v1, v0
    stp       q29, q31, [x1, #0x20]
    add       x1, x1, #0x80
    mov.16b   v19, v9
    mov.16b   v21, v10
    mov.16b   v23, v11
    mov.16b   v24, v13
    mov.16b   v13, v3
    mov.16b   v26, v14
    mov.16b   v14, v2
    mov.16b   v28, v15
    mov.16b   v0, v30
    ldr       q30, [sp, #0x80]
    mov.16b   v1, v8
    subs      x17, x17, #0x1
    b.ne      0x1000ed910

// ---- lanes f32 squared, W=32: 61 insns per row (1.91 per cell), 16 min instructions, 1 stack access(es)
LOOP 1000f02c0-1000f03b0: 61 insns, calls/traps []
    mov.16b   v15, v26
    mov.16b   v27, v8
    mov.16b   v1, v28
    mov.16b   v0, v29
    mov.16b   v2, v30
    mov.16b   v14, v31
    ld1r.4s   { v12 }, [x4], #4
    mov.16b   v13, v9
    mov.16b   v11, v10
    ldp       q26, q8, [x5, #-0x40]
    fminnm.4s v28, v15, v26
    fminnm.4s v23, v28, v23
    ldr       q28, [sp, #0x110]
    fsub.4s   v28, v12, v28
    fmul.4s   v28, v28, v28
    fadd.4s   v23, v28, v23
    fsub.4s   v28, v12, v3
    fmul.4s   v28, v28, v28
    fminnm.4s v27, v27, v8
    fminnm.4s v20, v27, v20
    fadd.4s   v20, v28, v20
    stp       q23, q20, [x5, #-0x40]
    ldp       q28, q29, [x5, #-0x20]
    fminnm.4s v1, v1, v28
    fminnm.4s v1, v1, v22
    fsub.4s   v22, v12, v4
    fmul.4s   v22, v22, v22
    fadd.4s   v22, v22, v1
    fsub.4s   v1, v12, v5
    fmul.4s   v1, v1, v1
    fminnm.4s v0, v0, v29
    fminnm.4s v0, v0, v19
    fadd.4s   v19, v1, v0
    stp       q22, q19, [x5, #-0x20]
    ldp       q30, q31, [x5]
    fminnm.4s v0, v2, v30
    fminnm.4s v0, v0, v21
    fsub.4s   v1, v12, v6
    fmul.4s   v1, v1, v1
    fadd.4s   v21, v1, v0
    fsub.4s   v0, v12, v7
    fmul.4s   v0, v0, v0
    fminnm.4s v1, v14, v31
    fminnm.4s v1, v1, v18
    fadd.4s   v18, v0, v1
    stp       q21, q18, [x5]
    ldp       q9, q10, [x5, #0x20]
    fminnm.4s v0, v13, v9
    fminnm.4s v0, v0, v25
    fsub.4s   v1, v12, v16
    fmul.4s   v1, v1, v1
    fadd.4s   v25, v1, v0
    fsub.4s   v0, v12, v17
    fmul.4s   v0, v0, v0
    fminnm.4s v1, v11, v10
    fminnm.4s v1, v1, v24
    fadd.4s   v24, v0, v1
    stp       q25, q24, [x5, #0x20]
    add       x5, x5, #0x80
    subs      x3, x3, #0x1
    b.ne      0x1000f02c0

// ---- lanes f32 L1, W=32: 53 insns per row (1.66 per cell), 16 min instructions, 1 stack access(es)
LOOP 1000f2c04-1000f2cd4: 53 insns, calls/traps []
    mov.16b   v13, v26
    mov.16b   v14, v27
    mov.16b   v15, v28
    mov.16b   v8, v29
    mov.16b   v1, v30
    mov.16b   v0, v31
    mov.16b   v12, v9
    mov.16b   v11, v10
    ld1r.4s   { v2 }, [x4], #4
    ldp       q26, q27, [x5, #-0x40]
    fminnm.4s v28, v13, v26
    fminnm.4s v23, v28, v23
    ldr       q28, [sp, #0x110]
    fabd.4s   v28, v2, v28
    fadd.4s   v23, v28, v23
    fabd.4s   v28, v2, v3
    fminnm.4s v29, v14, v27
    fminnm.4s v20, v29, v20
    fadd.4s   v20, v28, v20
    stp       q23, q20, [x5, #-0x40]
    ldp       q28, q29, [x5, #-0x20]
    fminnm.4s v30, v15, v28
    fminnm.4s v22, v30, v22
    fabd.4s   v30, v2, v4
    fadd.4s   v22, v30, v22
    fabd.4s   v30, v2, v5
    fminnm.4s v31, v8, v29
    fminnm.4s v19, v31, v19
    fadd.4s   v19, v30, v19
    stp       q22, q19, [x5, #-0x20]
    ldp       q30, q31, [x5]
    fminnm.4s v1, v1, v30
    fminnm.4s v1, v1, v21
    fabd.4s   v21, v2, v6
    fadd.4s   v21, v21, v1
    fabd.4s   v1, v2, v7
    fminnm.4s v0, v0, v31
    fminnm.4s v0, v0, v18
    fadd.4s   v18, v1, v0
    stp       q21, q18, [x5]
    ldp       q9, q10, [x5, #0x20]
    fminnm.4s v0, v12, v9
    fminnm.4s v0, v0, v25
    fabd.4s   v1, v2, v16
    fadd.4s   v25, v1, v0
    fabd.4s   v0, v2, v17
    fminnm.4s v1, v11, v10
    fminnm.4s v1, v1, v24
    fadd.4s   v24, v0, v1
    stp       q25, q24, [x5, #0x20]
    add       x5, x5, #0x80
    subs      x3, x3, #0x1
    b.ne      0x1000f2c04

