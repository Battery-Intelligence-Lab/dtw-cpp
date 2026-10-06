// The lanes kernel (dtwc/core/dtw_kernel.hpp:460-469) inner loop, one row i of W lanes per iteration.
// Identical instruction for instruction (general-purpose register numbers aside, vector registers equal)
// in build/bin/dtwc_cl (-march=native), the wheel _dtwcpp_core.cpython-312-darwin.so (no -march),
// build-matlab dtwc_mex.mexmaca64 (-march=native) and the kbench probe (probes/normregs.py diff).
// Addresses: dtwc_cl / wheel / MEX.

// ---- f64 L1 (W=8: 4 x .2d), 36 insns   [1000ecd8c-1000ece18 / c6dd8-c6e64 / 31edfc-31ee88]
LOOP 1000ecd8c-1000ece18: 36 insns, calls/traps []
    ldur      q21, [x1, #-0x20]
    ld1r.2d   { v22 }, [x0], #8
    fabd.2d   v23, v22, v1
    fcmgt.2d  v24, v7, v21
    bit.16b   v7, v21, v24
    fcmgt.2d  v24, v7, v5
    bif.16b   v5, v7, v24
    fadd.2d   v5, v5, v23
    ldp       q23, q24, [x1, #-0x10]
    fcmgt.2d  v7, v16, v23
    bsl.16b   v7, v23, v16
    fcmgt.2d  v16, v7, v6
    bif.16b   v6, v7, v16
    fabd.2d   v7, v22, v2
    fadd.2d   v6, v6, v7
    stp       q5, q6, [x1, #-0x20]
    fabd.2d   v7, v22, v3
    fcmgt.2d  v16, v18, v24
    bsl.16b   v16, v24, v18
    fcmgt.2d  v18, v16, v17
    bit.16b   v16, v17, v18
    fadd.2d   v17, v16, v7
    ldr       q25, [x1, #0x10]
    fabd.2d   v7, v22, v4
    fcmgt.2d  v16, v19, v25
    bsl.16b   v16, v25, v19
    fcmgt.2d  v18, v16, v20
    bit.16b   v16, v20, v18
    fadd.2d   v20, v16, v7
    stp       q17, q20, [x1], #0x40
    mov.16b   v7, v21
    mov.16b   v16, v23
    mov.16b   v18, v24
    mov.16b   v19, v25
    subs      x17, x17, #0x1
    b.ne      0x1000ecd8c


// ---- f64 squared (W=8), 40 insns   [1000ec020-1000ec0bc / c606c-c6108 / 31e090-31e12c]
LOOP 1000ec020-1000ec0bc: 40 insns, calls/traps []
    ldur      q21, [x1, #-0x20]
    ld1r.2d   { v22 }, [x0], #8
    fsub.2d   v23, v22, v1
    fmul.2d   v23, v23, v23
    fcmgt.2d  v24, v7, v21
    bit.16b   v7, v21, v24
    fcmgt.2d  v24, v7, v5
    bif.16b   v5, v7, v24
    fadd.2d   v5, v5, v23
    ldp       q23, q24, [x1, #-0x10]
    fcmgt.2d  v7, v16, v23
    bsl.16b   v7, v23, v16
    fcmgt.2d  v16, v7, v6
    bif.16b   v6, v7, v16
    fsub.2d   v7, v22, v2
    fmul.2d   v7, v7, v7
    fadd.2d   v6, v6, v7
    stp       q5, q6, [x1, #-0x20]
    fsub.2d   v7, v22, v3
    fmul.2d   v7, v7, v7
    fcmgt.2d  v16, v18, v24
    bsl.16b   v16, v24, v18
    fcmgt.2d  v18, v16, v17
    bit.16b   v16, v17, v18
    fadd.2d   v17, v16, v7
    ldr       q25, [x1, #0x10]
    fsub.2d   v7, v22, v4
    fmul.2d   v7, v7, v7
    fcmgt.2d  v16, v19, v25
    bsl.16b   v16, v25, v19
    fcmgt.2d  v18, v16, v20
    bit.16b   v16, v20, v18
    fadd.2d   v20, v16, v7
    stp       q17, q20, [x1], #0x40
    mov.16b   v7, v21
    mov.16b   v16, v23
    mov.16b   v18, v24
    mov.16b   v19, v25
    subs      x17, x17, #0x1
    b.ne      0x1000ec020


// ---- f32 L1 (W=16: 4 x .4s), 36 insns   [1000f0310-1000f039c / ca35c-ca3e8 / 322380-32240c]
LOOP 1000f0310-1000f039c: 36 insns, calls/traps []
    ldur      q22, [x0, #-0x20]
    ld1r.4s   { v23 }, [x17], #4
    fabd.4s   v24, v23, v2
    fcmgt.4s  v25, v7, v22
    bit.16b   v7, v22, v25
    fcmgt.4s  v25, v7, v6
    bif.16b   v6, v7, v25
    fadd.4s   v6, v6, v24
    ldp       q24, q25, [x0, #-0x10]
    fcmgt.4s  v7, v16, v24
    bsl.16b   v7, v24, v16
    fcmgt.4s  v16, v7, v17
    bit.16b   v7, v17, v16
    fabd.4s   v16, v23, v3
    fadd.4s   v17, v7, v16
    stp       q6, q17, [x0, #-0x20]
    fabd.4s   v7, v23, v4
    fcmgt.4s  v16, v18, v25
    bsl.16b   v16, v25, v18
    fcmgt.4s  v18, v16, v19
    bit.16b   v16, v19, v18
    fadd.4s   v19, v16, v7
    ldr       q26, [x0, #0x10]
    fabd.4s   v7, v23, v5
    fcmgt.4s  v16, v20, v26
    bsl.16b   v16, v26, v20
    fcmgt.4s  v18, v16, v21
    bit.16b   v16, v21, v18
    fadd.4s   v21, v16, v7
    stp       q19, q21, [x0], #0x40
    mov.16b   v7, v22
    mov.16b   v16, v24
    mov.16b   v18, v25
    mov.16b   v20, v26
    subs      x16, x16, #0x1
    b.ne      0x1000f0310


// ---- f32 squared (W=16), 40 insns   [1000ee078-1000ee114 / c80c4-c8160 / 3200e8-320184]
LOOP 1000ee078-1000ee114: 40 insns, calls/traps []
    ldur      q22, [x0, #-0x20]
    ld1r.4s   { v23 }, [x17], #4
    fsub.4s   v24, v23, v2
    fmul.4s   v24, v24, v24
    fcmgt.4s  v25, v7, v22
    bit.16b   v7, v22, v25
    fcmgt.4s  v25, v7, v6
    bif.16b   v6, v7, v25
    fadd.4s   v6, v6, v24
    ldp       q24, q25, [x0, #-0x10]
    fcmgt.4s  v7, v16, v24
    bsl.16b   v7, v24, v16
    fcmgt.4s  v16, v7, v17
    bit.16b   v7, v17, v16
    fsub.4s   v16, v23, v3
    fmul.4s   v16, v16, v16
    fadd.4s   v17, v7, v16
    stp       q6, q17, [x0, #-0x20]
    fsub.4s   v7, v23, v4
    fmul.4s   v7, v7, v7
    fcmgt.4s  v16, v18, v25
    bsl.16b   v16, v25, v18
    fcmgt.4s  v18, v16, v19
    bit.16b   v16, v19, v18
    fadd.4s   v19, v16, v7
    ldr       q26, [x0, #0x10]
    fsub.4s   v7, v23, v5
    fmul.4s   v7, v7, v7
    fcmgt.4s  v16, v20, v26
    bsl.16b   v16, v26, v20
    fcmgt.4s  v18, v16, v21
    bit.16b   v16, v21, v18
    fadd.4s   v21, v16, v7
    stp       q19, q21, [x0], #0x40
    mov.16b   v7, v22
    mov.16b   v16, v24
    mov.16b   v18, v25
    mov.16b   v20, v26
    subs      x16, x16, #0x1
    b.ne      0x1000ee078


