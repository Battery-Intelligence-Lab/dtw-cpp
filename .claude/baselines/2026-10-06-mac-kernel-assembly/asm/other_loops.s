// lb_keogh (dtwc/core/lower_bound_impl.hpp:173-197), dtwc_cl TADPole .omp_outlined: 2 x .2d x 4 per iteration, fmaxnm, no call or trap
LOOP 10009570c-100095788: 32 insns, calls/traps []
    ldp       q5, q6, [x16, #-0x20]
    ldp       q7, q16, [x16], #0x40
    ldp       q17, q18, [x14, #-0x20]
    ldp       q19, q20, [x14], #0x40
    fsub.2d   v17, v5, v17
    fsub.2d   v18, v6, v18
    fsub.2d   v19, v7, v19
    fsub.2d   v20, v16, v20
    ldp       q21, q22, [x15, #-0x20]
    ldp       q23, q24, [x15], #0x40
    fsub.2d   v5, v21, v5
    fsub.2d   v6, v22, v6
    fsub.2d   v7, v23, v7
    fsub.2d   v16, v24, v16
    fmaxnm.2d v17, v17, v25
    fmaxnm.2d v18, v18, v25
    fmaxnm.2d v19, v19, v25
    fmaxnm.2d v20, v20, v25
    fmaxnm.2d v5, v5, v25
    fmaxnm.2d v6, v6, v25
    fmaxnm.2d v7, v7, v25
    fmaxnm.2d v16, v16, v25
    fadd.2d   v1, v17, v1
    fadd.2d   v2, v18, v2
    fadd.2d   v3, v19, v3
    fadd.2d   v4, v20, v4
    fadd.2d   v1, v1, v5
    fadd.2d   v2, v2, v6
    fadd.2d   v3, v3, v7
    fadd.2d   v4, v4, v16
    subs      x17, x17, #0x8
    b.ne      0x10009570c

// z_normalize (dtwc/core/z_normalize.hpp:42-77) in the wheel: the binding TU is -Os (build log line 538), so the three passes are
// tail-folded 2-lane loops with a per-lane mask test and lane loads (cmhs/xtn/tbz/ld1) - @21744, @217a8, @2182c
LOOP 21744-21778: 14 insns, calls/traps []
    mov.16b   v3, v5
    cmhs.2d   v4, v0, v1
    xtn.2s    v6, v4
    fmov      w13, s6
    tbz       w13, #0x0, 0x2175c
    ldur      d5, [x11, #-0x8]
    mov.s     w13, v6[1]
    tbz       w13, #0x0, 0x21768
    ld1.d     { v5 }[1], [x11]
    fadd.2d   v5, v5, v3
    add.2d    v1, v1, v2
    add       x11, x11, #0x10
    subs      x12, x12, #0x2
    b.ne      0x21744
LOOP 217a8-217e4: 16 insns, calls/traps []
    mov.16b   v6, v16
    cmhs.2d   v7, v0, v4
    xtn.2s    v17, v7
    fmov      w12, s17
    tbz       w12, #0x0, 0x217c0
    ldur      d16, [x11, #-0x8]
    mov.s     w12, v17[1]
    tbz       w12, #0x0, 0x217cc
    ld1.d     { v16 }[1], [x11]
    fsub.2d   v17, v16, v3
    mov.16b   v16, v6
    fmla.2d   v16, v17, v17
    add.2d    v4, v4, v5
    add       x11, x11, #0x10
    subs      x10, x10, #0x2
    b.ne      0x217a8
LOOP 2182c-21870: 18 insns, calls/traps []
    cmhs.2d   v5, v0, v2
    xtn.2s    v5, v5
    fmov      w9, s5
    tbz       w9, #0x0, 0x2184c
    ldur      d6, [x10, #-0x8]
    fsub      d6, d6, d1
    fmul      d6, d6, d4
    stur      d6, [x10, #-0x8]
    mov.s     w9, v5[1]
    tbz       w9, #0x0, 0x21864
    ldr       d5, [x10]
    fsub      d5, d5, d1
    fmul      d5, d5, d4
    str       d5, [x10]
    add.2d    v2, v2, v3
    add       x10, x10, #0x10
    subs      x8, x8, #0x2
    b.ne      0x2182c

// pass 2 of z_normalize after relinking the binding TU without -Os (scratch nominsize/...so @2a658): 4 x .2d fmla per iteration
LOOP 2a658-2a684: 12 insns, calls/traps []
    ldp       q7, q16, [x10, #-0x20]
    ldp       q17, q18, [x10], #0x40
    fsub.2d   v7, v7, v2
    fsub.2d   v16, v16, v2
    fsub.2d   v17, v17, v2
    fsub.2d   v18, v18, v2
    fmla.2d   v3, v7, v7
    fmla.2d   v4, v16, v16
    fmla.2d   v5, v17, v17
    fmla.2d   v6, v18, v18
    subs      x11, x11, #0x8
    b.ne      0x2a658

// FasterPAM find_best_swap (fast_pam.cpp:145-158), inlined in dtwc::fast_pam_seeded, dtwc_cl @1000880c8-10008812c: no call, no spill;
// tri_index per element (cmp/csel/csel/madd/lsr/add), acc += doj - d1 reassociated to (acc + doj) - d1 (two ops on the acc chain)
1000880c0:     	ldur	x12, [x29, #-0xc0]
1000880c4:     	b	0x1000880f4 
1000880c8:     	fadd	d0, d1, d0
1000880cc:     	fsub	d0, d0, d2
1000880d0:     	ldr	d1, [x11, x8, lsl #3]
1000880d4:     	fsub	d1, d2, d1
1000880d8:     	ldr	x13, [x12, x8, lsl #3]
1000880dc:     	ldr	d2, [x26, x13, lsl #3]
1000880e0:     	fadd	d1, d2, d1
1000880e4:     	str	d1, [x26, x13, lsl #3]
1000880e8:     	add	x8, x8, #0x1
1000880ec:     	cmp	x24, x8
1000880f0:     	b.eq	0x100088130 
1000880f4:     	cmp	x21, x8
1000880f8:     	csel	x13, x21, x8, hi
1000880fc:     	csel	x14, x21, x8, lo
100088100:     	madd	x13, x13, x13, x13
100088104:     	lsr	x13, x13, #1
100088108:     	add	x13, x9, x13, lsl #3
10008810c:     	ldr	d1, [x13, x14, lsl #3]
100088110:     	ldr	d2, [x10, x8, lsl #3]
100088114:     	fcmp	d1, d2
100088118:     	b.mi	0x1000880c8 
10008811c:     	ldr	d2, [x11, x8, lsl #3]
100088120:     	fcmp	d1, d2
100088124:     	b.pl	0x1000880e8 
100088128:     	fsub	d1, d1, d2
10008812c:     	b	0x1000880d8 

// FasterPAM compute_nearest_and_second OpenMP body (fast_pam.cpp:83-99), dtwc_cl @100086c08: no call, no spill (bl in the outer loop is __kmpc_dispatch_next_8)
INNER loop 100086c08-100086c50: 19 insns, vector-operand insns 0, calls/traps []
    100086c08: ldr      x2, [x16, x1, lsl #3]
    100086c0c: cmp      x8, x2
    100086c10: csel     x3, x8, x2, hi
    100086c14: csel     x4, x8, x2, lo
    100086c18: madd     x3, x3, x3, x3
    100086c1c: lsr      x3, x3, #1
    100086c20: add      x3, x17, x3, lsl #3
    100086c24: ldr      d2, [x3, x4, lsl #3]
    100086c28: fcmp     d2, d0
    100086c2c: b.mi     0x100086bf0
    100086c30: cmp      x2, x8
    100086c34: b.ne     0x100086c40
    100086c38: fcmp     d2, d0
    100086c3c: b.eq     0x100086bf0
    100086c40: fcmp     d2, d1
    100086c44: fcsel    d1, d2, d1, mi
    100086c48: add      x1, x1, #0x1
    100086c4c: cmp      x15, x1
    100086c50: b.ne     0x100086c08
