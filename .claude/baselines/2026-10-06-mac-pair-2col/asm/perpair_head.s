// Per-pair Standard loops of the linked dtwc_cl at head (kernels 1 and 2 two columns per pass): the pair loop, then the one-column loop; dtw_banded<false>'s first listed loop is its per-pass skeleton (rows 0 and the one-column rows of a pass, no inner loop), not a per-cell loop
// ---- kernel 1 (linear) f64 L1
LOOP 1000ada58-1000adaa8 21 insns [ldr:4 str:1 fabd:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000ada58: ldr       d5, [x10]
    1000ada5c: ldr       d6, [x15], #T
    1000ada60: ldr       d7, [x21, x13, lsl #3]
    1000ada64: fabd      d7, d6, d7
    1000ada68: fcmp      d5, d4
    1000ada6c: fcsel     d4, d5, d4, mi
    1000ada70: fcmp      d3, d4
    1000ada74: fcsel     d4, d3, d4, mi
    1000ada78: ldr       d16, [x21, x14, lsl #3]
    1000ada7c: fadd      d7, d4, d7
    1000ada80: fabd      d4, d6, d16
    1000ada84: fcmp      d7, d3
    1000ada88: fcsel     d3, d7, d3, mi
    1000ada8c: fcmp      d2, d3
    1000ada90: fcsel     d2, d2, d3, mi
    1000ada94: fadd      d2, d2, d4
    1000ada98: str       d2, [x10], #T
    1000ada9c: mov.16b   v4, v5
    1000adaa0: mov.16b   v3, v7
    1000adaa4: subs      x16, x16, #T
    1000adaa8: b.ne      T

// ---- kernel 1 (linear) f64 L1
LOOP 1000adb58-1000adb88 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000adb58: ldr       d4, [x13]
    1000adb5c: ldr       d5, [x14], #T
    1000adb60: ldr       d6, [x21, x10, lsl #3]
    1000adb64: fabd      d5, d5, d6
    1000adb68: fcmp      d4, d3
    1000adb6c: fcsel     d3, d4, d3, mi
    1000adb70: fcmp      d2, d3
    1000adb74: fcsel     d2, d2, d3, mi
    1000adb78: fadd      d2, d2, d5
    1000adb7c: str       d2, [x13], #T
    1000adb80: mov.16b   v3, v4
    1000adb84: subs      x15, x15, #T
    1000adb88: b.ne      T

// ---- kernel 2 (banded) f64 L1
LOOP 1000ad6c4-1000ad724 25 insns [ldr:6 str:2 fabd:2 fcmp:2 fcsel:2 fadd:2 fminnm:1 mov:1] calls/traps []
    1000ad6c4: ldr       d4, [x2]
    1000ad6c8: ldr       d3, [x9, x15, lsl #3]
    1000ad6cc: ldr       d5, [x10, x3, lsl #3]
    1000ad6d0: fabd      d3, d3, d5
    1000ad6d4: fcmp      d4, d2
    1000ad6d8: fcsel     d2, d4, d2, mi
    1000ad6dc: fcmp      d1, d2
    1000ad6e0: fcsel     d1, d1, d2, mi
    1000ad6e4: fadd      d1, d1, d3
    1000ad6e8: str       d1, [x2]
    1000ad6ec: mov       x2, #T
    1000ad6f0: fmov      d3, x2
    1000ad6f4: mov.16b   v2, v4
    1000ad6f8: subs      x2, x17, x1
    1000ad6fc: b.hi      T
    1000ad700: b         T
    1000ad704: ldr       d1, [x9, x15, lsl #3]
    1000ad708: ldr       d3, [x8]
    1000ad70c: ldr       d4, [x10]
    1000ad710: fabd      d1, d1, d4
    1000ad714: fminnm    d3, d3, d0
    1000ad718: fadd      d1, d1, d3
    1000ad71c: str       d1, [x8]
    1000ad720: cmp       x4, x3
    1000ad724: b.hi      T

// ---- kernel 2 (banded) f64 L1
LOOP 1000ad748-1000ad798 21 insns [ldr:4 str:1 fabd:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000ad748: ldr       d5, [x3]
    1000ad74c: ldr       d4, [x9, x15, lsl #3]
    1000ad750: ldr       d6, [x1], #T
    1000ad754: fabd      d4, d4, d6
    1000ad758: fcmp      d5, d2
    1000ad75c: fcsel     d2, d5, d2, mi
    1000ad760: fcmp      d1, d2
    1000ad764: fcsel     d2, d1, d2, mi
    1000ad768: ldr       d7, [x9, x16, lsl #3]
    1000ad76c: fadd      d4, d2, d4
    1000ad770: fabd      d2, d7, d6
    1000ad774: fcmp      d4, d1
    1000ad778: fcsel     d1, d4, d1, mi
    1000ad77c: fcmp      d3, d1
    1000ad780: fcsel     d1, d3, d1, mi
    1000ad784: fadd      d3, d1, d2
    1000ad788: str       d3, [x3], #T
    1000ad78c: mov.16b   v2, v5
    1000ad790: mov.16b   v1, v4
    1000ad794: subs      x2, x2, #T
    1000ad798: b.ne      T

// ---- kernel 2 (banded) f64 L1
LOOP 1000ad8bc-1000ad8ec 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad8bc: ldr       d2, [x12]
    1000ad8c0: ldr       d3, [x9, x17, lsl #3]
    1000ad8c4: ldr       d4, [x10], #T
    1000ad8c8: fabd      d3, d3, d4
    1000ad8cc: fcmp      d2, d1
    1000ad8d0: fcsel     d1, d2, d1, mi
    1000ad8d4: fcmp      d0, d1
    1000ad8d8: fcsel     d0, d0, d1, mi
    1000ad8dc: fadd      d0, d0, d3
    1000ad8e0: str       d0, [x12], #T
    1000ad8e4: mov.16b   v1, v2
    1000ad8e8: subs      x11, x11, #T
    1000ad8ec: b.ne      T

// ---- kernel 1 (linear) f64 squared
LOOP 1000ad32c-1000ad384 23 insns [ldr:4 str:1 fsub:2 fmul:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000ad32c: ldr       d5, [x15], #T
    1000ad330: ldr       d6, [x11]
    1000ad334: ldr       d7, [x21, x13, lsl #3]
    1000ad338: fsub      d7, d5, d7
    1000ad33c: fmul      d7, d7, d7
    1000ad340: fcmp      d6, d4
    1000ad344: fcsel     d4, d6, d4, mi
    1000ad348: fcmp      d3, d4
    1000ad34c: fcsel     d4, d3, d4, mi
    1000ad350: fadd      d7, d4, d7
    1000ad354: ldr       d4, [x21, x14, lsl #3]
    1000ad358: fsub      d4, d5, d4
    1000ad35c: fmul      d4, d4, d4
    1000ad360: fcmp      d7, d3
    1000ad364: fcsel     d3, d7, d3, mi
    1000ad368: fcmp      d2, d3
    1000ad36c: fcsel     d2, d2, d3, mi
    1000ad370: fadd      d2, d2, d4
    1000ad374: str       d2, [x11], #T
    1000ad378: mov.16b   v4, v6
    1000ad37c: mov.16b   v3, v7
    1000ad380: subs      x16, x16, #T
    1000ad384: b.ne      T

// ---- kernel 1 (linear) f64 squared
LOOP 1000ad43c-1000ad470 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad43c: ldr       d4, [x14], #T
    1000ad440: ldr       d5, [x21, x11, lsl #3]
    1000ad444: ldr       d6, [x13]
    1000ad448: fsub      d4, d4, d5
    1000ad44c: fmul      d4, d4, d4
    1000ad450: fcmp      d6, d3
    1000ad454: fcsel     d3, d6, d3, mi
    1000ad458: fcmp      d2, d3
    1000ad45c: fcsel     d2, d2, d3, mi
    1000ad460: fadd      d2, d2, d4
    1000ad464: str       d2, [x13], #T
    1000ad468: mov.16b   v3, v6
    1000ad46c: subs      x15, x15, #T
    1000ad470: b.ne      T

// ---- kernel 2 (banded) f64 squared
LOOP 1000acf68-1000acfd0 27 insns [ldr:6 str:2 fsub:2 fmul:2 fcmp:2 fcsel:2 fadd:2 fminnm:1 mov:1] calls/traps []
    1000acf68: ldr       d3, [x9, x15, lsl #3]
    1000acf6c: ldr       d4, [x10, x3, lsl #3]
    1000acf70: ldr       d5, [x2]
    1000acf74: fsub      d3, d3, d4
    1000acf78: fmul      d3, d3, d3
    1000acf7c: fcmp      d5, d2
    1000acf80: fcsel     d2, d5, d2, mi
    1000acf84: fcmp      d1, d2
    1000acf88: fcsel     d1, d1, d2, mi
    1000acf8c: fadd      d1, d1, d3
    1000acf90: str       d1, [x2]
    1000acf94: mov       x2, #T
    1000acf98: fmov      d3, x2
    1000acf9c: mov.16b   v2, v5
    1000acfa0: subs      x2, x17, x1
    1000acfa4: b.hi      T
    1000acfa8: b         T
    1000acfac: ldr       d1, [x8]
    1000acfb0: ldr       d3, [x9, x15, lsl #3]
    1000acfb4: ldr       d4, [x10]
    1000acfb8: fsub      d3, d3, d4
    1000acfbc: fmul      d3, d3, d3
    1000acfc0: fminnm    d1, d1, d0
    1000acfc4: fadd      d1, d3, d1
    1000acfc8: str       d1, [x8]
    1000acfcc: cmp       x4, x3
    1000acfd0: b.hi      T

// ---- kernel 2 (banded) f64 squared
LOOP 1000acff4-1000ad04c 23 insns [ldr:4 str:1 fsub:2 fmul:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000acff4: ldr       d4, [x9, x15, lsl #3]
    1000acff8: ldr       d5, [x3]
    1000acffc: ldr       d6, [x1], #T
    1000ad000: fsub      d4, d4, d6
    1000ad004: fmul      d4, d4, d4
    1000ad008: fcmp      d5, d2
    1000ad00c: fcsel     d2, d5, d2, mi
    1000ad010: fcmp      d1, d2
    1000ad014: fcsel     d2, d1, d2, mi
    1000ad018: fadd      d4, d2, d4
    1000ad01c: ldr       d2, [x9, x16, lsl #3]
    1000ad020: fsub      d2, d2, d6
    1000ad024: fmul      d2, d2, d2
    1000ad028: fcmp      d4, d1
    1000ad02c: fcsel     d1, d4, d1, mi
    1000ad030: fcmp      d3, d1
    1000ad034: fcsel     d1, d3, d1, mi
    1000ad038: fadd      d3, d1, d2
    1000ad03c: str       d3, [x3], #T
    1000ad040: mov.16b   v2, v5
    1000ad044: mov.16b   v1, v4
    1000ad048: subs      x2, x2, #T
    1000ad04c: b.ne      T

// ---- kernel 2 (banded) f64 squared
LOOP 1000ad17c-1000ad1b0 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad17c: ldr       d2, [x9, x17, lsl #3]
    1000ad180: ldr       d3, [x10], #T
    1000ad184: ldr       d4, [x12]
    1000ad188: fsub      d2, d2, d3
    1000ad18c: fmul      d2, d2, d2
    1000ad190: fcmp      d4, d0
    1000ad194: fcsel     d0, d4, d0, mi
    1000ad198: fcmp      d1, d0
    1000ad19c: fcsel     d0, d1, d0, mi
    1000ad1a0: fadd      d1, d0, d2
    1000ad1a4: str       d1, [x12], #T
    1000ad1a8: mov.16b   v0, v4
    1000ad1ac: subs      x11, x11, #T
    1000ad1b0: b.ne      T

// ---- kernel 1 (linear) f32 L1
LOOP 1000c2738-1000c2788 21 insns [ldr:4 str:1 fabd:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000c2738: ldr       s5, [x10]
    1000c273c: ldr       s6, [x15], #T
    1000c2740: ldr       s7, [x21, x13, lsl #2]
    1000c2744: fabd      s7, s6, s7
    1000c2748: fcmp      s5, s4
    1000c274c: fcsel     s4, s5, s4, mi
    1000c2750: fcmp      s3, s4
    1000c2754: fcsel     s4, s3, s4, mi
    1000c2758: ldr       s16, [x21, x14, lsl #2]
    1000c275c: fadd      s7, s4, s7
    1000c2760: fabd      s4, s6, s16
    1000c2764: fcmp      s7, s3
    1000c2768: fcsel     s3, s7, s3, mi
    1000c276c: fcmp      s2, s3
    1000c2770: fcsel     s2, s2, s3, mi
    1000c2774: fadd      s2, s2, s4
    1000c2778: str       s2, [x10], #T
    1000c277c: mov.16b   v4, v5
    1000c2780: mov.16b   v3, v7
    1000c2784: subs      x16, x16, #T
    1000c2788: b.ne      T

// ---- kernel 1 (linear) f32 L1
LOOP 1000c2838-1000c2868 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000c2838: ldr       s4, [x13]
    1000c283c: ldr       s5, [x14], #T
    1000c2840: ldr       s6, [x21, x10, lsl #2]
    1000c2844: fabd      s5, s5, s6
    1000c2848: fcmp      s4, s3
    1000c284c: fcsel     s3, s4, s3, mi
    1000c2850: fcmp      s2, s3
    1000c2854: fcsel     s2, s2, s3, mi
    1000c2858: fadd      s2, s2, s5
    1000c285c: str       s2, [x13], #T
    1000c2860: mov.16b   v3, v4
    1000c2864: subs      x15, x15, #T
    1000c2868: b.ne      T

// ---- kernel 2 (banded) f32 L1
LOOP 1000c23a8-1000c2408 25 insns [ldr:6 str:2 fabd:2 fcmp:2 fcsel:2 fadd:2 fminnm:1 mov:1] calls/traps []
    1000c23a8: ldr       s4, [x2]
    1000c23ac: ldr       s3, [x9, x15, lsl #2]
    1000c23b0: ldr       s5, [x10, x3, lsl #2]
    1000c23b4: fabd      s3, s3, s5
    1000c23b8: fcmp      s4, s2
    1000c23bc: fcsel     s2, s4, s2, mi
    1000c23c0: fcmp      s1, s2
    1000c23c4: fcsel     s1, s1, s2, mi
    1000c23c8: fadd      s1, s1, s3
    1000c23cc: str       s1, [x2]
    1000c23d0: mov       w2, #T
    1000c23d4: fmov      s3, w2
    1000c23d8: mov.16b   v2, v4
    1000c23dc: subs      x2, x17, x1
    1000c23e0: b.hi      T
    1000c23e4: b         T
    1000c23e8: ldr       s1, [x9, x15, lsl #2]
    1000c23ec: ldr       s3, [x8]
    1000c23f0: ldr       s4, [x10]
    1000c23f4: fabd      s1, s1, s4
    1000c23f8: fminnm    s3, s3, s0
    1000c23fc: fadd      s1, s1, s3
    1000c2400: str       s1, [x8]
    1000c2404: cmp       x4, x3
    1000c2408: b.hi      T

// ---- kernel 2 (banded) f32 L1
LOOP 1000c242c-1000c247c 21 insns [ldr:4 str:1 fabd:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000c242c: ldr       s5, [x3]
    1000c2430: ldr       s4, [x9, x15, lsl #2]
    1000c2434: ldr       s6, [x1], #T
    1000c2438: fabd      s4, s4, s6
    1000c243c: fcmp      s5, s2
    1000c2440: fcsel     s2, s5, s2, mi
    1000c2444: fcmp      s1, s2
    1000c2448: fcsel     s2, s1, s2, mi
    1000c244c: ldr       s7, [x9, x16, lsl #2]
    1000c2450: fadd      s4, s2, s4
    1000c2454: fabd      s2, s7, s6
    1000c2458: fcmp      s4, s1
    1000c245c: fcsel     s1, s4, s1, mi
    1000c2460: fcmp      s3, s1
    1000c2464: fcsel     s1, s3, s1, mi
    1000c2468: fadd      s3, s1, s2
    1000c246c: str       s3, [x3], #T
    1000c2470: mov.16b   v2, v5
    1000c2474: mov.16b   v1, v4
    1000c2478: subs      x2, x2, #T
    1000c247c: b.ne      T

// ---- kernel 2 (banded) f32 L1
LOOP 1000c259c-1000c25cc 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000c259c: ldr       s2, [x12]
    1000c25a0: ldr       s3, [x9, x17, lsl #2]
    1000c25a4: ldr       s4, [x10], #T
    1000c25a8: fabd      s3, s3, s4
    1000c25ac: fcmp      s2, s1
    1000c25b0: fcsel     s1, s2, s1, mi
    1000c25b4: fcmp      s0, s1
    1000c25b8: fcsel     s0, s0, s1, mi
    1000c25bc: fadd      s0, s0, s3
    1000c25c0: str       s0, [x12], #T
    1000c25c4: mov.16b   v1, v2
    1000c25c8: subs      x11, x11, #T
    1000c25cc: b.ne      T

// ---- kernel 1 (linear) f32 squared
LOOP 1000c2018-1000c2070 23 insns [ldr:4 str:1 fsub:2 fmul:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000c2018: ldr       s5, [x15], #T
    1000c201c: ldr       s6, [x11]
    1000c2020: ldr       s7, [x21, x13, lsl #2]
    1000c2024: fsub      s7, s5, s7
    1000c2028: fmul      s7, s7, s7
    1000c202c: fcmp      s6, s4
    1000c2030: fcsel     s4, s6, s4, mi
    1000c2034: fcmp      s3, s4
    1000c2038: fcsel     s4, s3, s4, mi
    1000c203c: fadd      s7, s4, s7
    1000c2040: ldr       s4, [x21, x14, lsl #2]
    1000c2044: fsub      s4, s5, s4
    1000c2048: fmul      s4, s4, s4
    1000c204c: fcmp      s7, s3
    1000c2050: fcsel     s3, s7, s3, mi
    1000c2054: fcmp      s2, s3
    1000c2058: fcsel     s2, s2, s3, mi
    1000c205c: fadd      s2, s2, s4
    1000c2060: str       s2, [x11], #T
    1000c2064: mov.16b   v4, v6
    1000c2068: mov.16b   v3, v7
    1000c206c: subs      x16, x16, #T
    1000c2070: b.ne      T

// ---- kernel 1 (linear) f32 squared
LOOP 1000c2128-1000c215c 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000c2128: ldr       s4, [x14], #T
    1000c212c: ldr       s5, [x21, x11, lsl #2]
    1000c2130: ldr       s6, [x13]
    1000c2134: fsub      s4, s4, s5
    1000c2138: fmul      s4, s4, s4
    1000c213c: fcmp      s6, s3
    1000c2140: fcsel     s3, s6, s3, mi
    1000c2144: fcmp      s2, s3
    1000c2148: fcsel     s2, s2, s3, mi
    1000c214c: fadd      s2, s2, s4
    1000c2150: str       s2, [x13], #T
    1000c2154: mov.16b   v3, v6
    1000c2158: subs      x15, x15, #T
    1000c215c: b.ne      T

// ---- kernel 2 (banded) f32 squared
LOOP 1000c1998-1000c1a00 27 insns [ldr:6 str:2 fsub:2 fmul:2 fcmp:2 fcsel:2 fadd:2 fminnm:1 mov:1] calls/traps []
    1000c1998: ldr       s3, [x9, x15, lsl #2]
    1000c199c: ldr       s4, [x10, x3, lsl #2]
    1000c19a0: ldr       s5, [x2]
    1000c19a4: fsub      s3, s3, s4
    1000c19a8: fmul      s3, s3, s3
    1000c19ac: fcmp      s5, s2
    1000c19b0: fcsel     s2, s5, s2, mi
    1000c19b4: fcmp      s1, s2
    1000c19b8: fcsel     s1, s1, s2, mi
    1000c19bc: fadd      s1, s1, s3
    1000c19c0: str       s1, [x2]
    1000c19c4: mov       w2, #T
    1000c19c8: fmov      s3, w2
    1000c19cc: mov.16b   v2, v5
    1000c19d0: subs      x2, x17, x1
    1000c19d4: b.hi      T
    1000c19d8: b         T
    1000c19dc: ldr       s1, [x8]
    1000c19e0: ldr       s3, [x9, x15, lsl #2]
    1000c19e4: ldr       s4, [x10]
    1000c19e8: fsub      s3, s3, s4
    1000c19ec: fmul      s3, s3, s3
    1000c19f0: fminnm    s1, s1, s0
    1000c19f4: fadd      s1, s3, s1
    1000c19f8: str       s1, [x8]
    1000c19fc: cmp       x4, x3
    1000c1a00: b.hi      T

// ---- kernel 2 (banded) f32 squared
LOOP 1000c1a24-1000c1a7c 23 insns [ldr:4 str:1 fsub:2 fmul:2 fcmp:4 fcsel:4 fadd:2] calls/traps []
    1000c1a24: ldr       s4, [x9, x15, lsl #2]
    1000c1a28: ldr       s5, [x3]
    1000c1a2c: ldr       s6, [x1], #T
    1000c1a30: fsub      s4, s4, s6
    1000c1a34: fmul      s4, s4, s4
    1000c1a38: fcmp      s5, s2
    1000c1a3c: fcsel     s2, s5, s2, mi
    1000c1a40: fcmp      s1, s2
    1000c1a44: fcsel     s2, s1, s2, mi
    1000c1a48: fadd      s4, s2, s4
    1000c1a4c: ldr       s2, [x9, x16, lsl #2]
    1000c1a50: fsub      s2, s2, s6
    1000c1a54: fmul      s2, s2, s2
    1000c1a58: fcmp      s4, s1
    1000c1a5c: fcsel     s1, s4, s1, mi
    1000c1a60: fcmp      s3, s1
    1000c1a64: fcsel     s1, s3, s1, mi
    1000c1a68: fadd      s3, s1, s2
    1000c1a6c: str       s3, [x3], #T
    1000c1a70: mov.16b   v2, v5
    1000c1a74: mov.16b   v1, v4
    1000c1a78: subs      x2, x2, #T
    1000c1a7c: b.ne      T

// ---- kernel 2 (banded) f32 squared
LOOP 1000c1ba8-1000c1bdc 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000c1ba8: ldr       s2, [x9, x17, lsl #2]
    1000c1bac: ldr       s3, [x10], #T
    1000c1bb0: ldr       s4, [x12]
    1000c1bb4: fsub      s2, s2, s3
    1000c1bb8: fmul      s2, s2, s2
    1000c1bbc: fcmp      s4, s0
    1000c1bc0: fcsel     s0, s4, s0, mi
    1000c1bc4: fcmp      s1, s0
    1000c1bc8: fcsel     s0, s1, s0, mi
    1000c1bcc: fadd      s1, s0, s2
    1000c1bd0: str       s1, [x12], #T
    1000c1bd4: mov.16b   v0, v4
    1000c1bd8: subs      x11, x11, #T
    1000c1bdc: b.ne      T

