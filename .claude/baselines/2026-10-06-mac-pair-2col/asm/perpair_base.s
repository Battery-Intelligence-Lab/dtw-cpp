// Per-pair Standard loops of the linked dtwc_cl at base e26d5680 (objdump -d; registers as linked; branch targets T)
// ---- kernel 1 (linear) f64 L1
LOOP 1000ad584-1000ad5b4 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad584: ldr       d4, [x13]
    1000ad588: ldr       d5, [x14], #T
    1000ad58c: ldr       d6, [x21, x12, lsl #3]
    1000ad590: fabd      d5, d5, d6
    1000ad594: fcmp      d4, d3
    1000ad598: fcsel     d3, d4, d3, mi
    1000ad59c: fcmp      d2, d3
    1000ad5a0: fcsel     d2, d2, d3, mi
    1000ad5a4: fadd      d2, d2, d5
    1000ad5a8: str       d2, [x13], #T
    1000ad5ac: mov.16b   v3, v4
    1000ad5b0: subs      x15, x15, #T
    1000ad5b4: b.ne      T

// ---- kernel 2 (banded) f64 L1
LOOP 1000ad3d0-1000ad400 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad3d0: ldr       d3, [x17]
    1000ad3d4: ldr       d4, [x9, x13, lsl #3]
    1000ad3d8: ldr       d5, [x16], #T
    1000ad3dc: fabd      d4, d4, d5
    1000ad3e0: fcmp      d3, d2
    1000ad3e4: fcsel     d2, d3, d2, mi
    1000ad3e8: fcmp      d1, d2
    1000ad3ec: fcsel     d1, d1, d2, mi
    1000ad3f0: fadd      d1, d1, d4
    1000ad3f4: str       d1, [x17], #T
    1000ad3f8: mov.16b   v2, v3
    1000ad3fc: subs      x15, x15, #T
    1000ad400: b.ne      T

// ---- kernel 1 (linear) f64 squared
LOOP 1000ad11c-1000ad150 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ad11c: ldr       d4, [x14], #T
    1000ad120: ldr       d5, [x21, x12, lsl #3]
    1000ad124: ldr       d6, [x13]
    1000ad128: fsub      d4, d4, d5
    1000ad12c: fmul      d4, d4, d4
    1000ad130: fcmp      d6, d2
    1000ad134: fcsel     d2, d6, d2, mi
    1000ad138: fcmp      d3, d2
    1000ad13c: fcsel     d2, d3, d2, mi
    1000ad140: fadd      d3, d2, d4
    1000ad144: str       d3, [x13], #T
    1000ad148: mov.16b   v2, v6
    1000ad14c: subs      x15, x15, #T
    1000ad150: b.ne      T

// ---- kernel 2 (banded) f64 squared
LOOP 1000acf58-1000acf8c 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000acf58: ldr       d3, [x9, x13, lsl #3]
    1000acf5c: ldr       d4, [x16], #T
    1000acf60: ldr       d5, [x17]
    1000acf64: fsub      d3, d3, d4
    1000acf68: fmul      d3, d3, d3
    1000acf6c: fcmp      d5, d2
    1000acf70: fcsel     d2, d5, d2, mi
    1000acf74: fcmp      d1, d2
    1000acf78: fcsel     d1, d1, d2, mi
    1000acf7c: fadd      d1, d1, d3
    1000acf80: str       d1, [x17], #T
    1000acf84: mov.16b   v2, v5
    1000acf88: subs      x15, x15, #T
    1000acf8c: b.ne      T

// ---- kernel 1 (linear) f32 L1
LOOP 1000bb0d8-1000bb108 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000bb0d8: ldr       s4, [x13]
    1000bb0dc: ldr       s5, [x14], #T
    1000bb0e0: ldr       s6, [x21, x12, lsl #2]
    1000bb0e4: fabd      s5, s5, s6
    1000bb0e8: fcmp      s4, s3
    1000bb0ec: fcsel     s3, s4, s3, mi
    1000bb0f0: fcmp      s2, s3
    1000bb0f4: fcsel     s2, s2, s3, mi
    1000bb0f8: fadd      s2, s2, s5
    1000bb0fc: str       s2, [x13], #T
    1000bb100: mov.16b   v3, v4
    1000bb104: subs      x15, x15, #T
    1000bb108: b.ne      T

// ---- kernel 2 (banded) f32 L1
LOOP 1000baf28-1000baf58 13 insns [ldr:3 str:1 fabd:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000baf28: ldr       s3, [x17]
    1000baf2c: ldr       s4, [x9, x13, lsl #2]
    1000baf30: ldr       s5, [x16], #T
    1000baf34: fabd      s4, s4, s5
    1000baf38: fcmp      s3, s2
    1000baf3c: fcsel     s2, s3, s2, mi
    1000baf40: fcmp      s1, s2
    1000baf44: fcsel     s1, s1, s2, mi
    1000baf48: fadd      s1, s1, s4
    1000baf4c: str       s1, [x17], #T
    1000baf50: mov.16b   v2, v3
    1000baf54: subs      x15, x15, #T
    1000baf58: b.ne      T

// ---- kernel 1 (linear) f32 squared
LOOP 1000ba9bc-1000ba9f0 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ba9bc: ldr       s4, [x14], #T
    1000ba9c0: ldr       s5, [x21, x12, lsl #2]
    1000ba9c4: ldr       s6, [x13]
    1000ba9c8: fsub      s4, s4, s5
    1000ba9cc: fmul      s4, s4, s4
    1000ba9d0: fcmp      s6, s2
    1000ba9d4: fcsel     s2, s6, s2, mi
    1000ba9d8: fcmp      s3, s2
    1000ba9dc: fcsel     s2, s3, s2, mi
    1000ba9e0: fadd      s3, s2, s4
    1000ba9e4: str       s3, [x13], #T
    1000ba9e8: mov.16b   v2, v6
    1000ba9ec: subs      x15, x15, #T
    1000ba9f0: b.ne      T

// ---- kernel 2 (banded) f32 squared
LOOP 1000ba7fc-1000ba830 14 insns [ldr:3 str:1 fsub:1 fmul:1 fcmp:2 fcsel:2 fadd:1] calls/traps []
    1000ba7fc: ldr       s3, [x9, x13, lsl #2]
    1000ba800: ldr       s4, [x16], #T
    1000ba804: ldr       s5, [x17]
    1000ba808: fsub      s3, s3, s4
    1000ba80c: fmul      s3, s3, s3
    1000ba810: fcmp      s5, s2
    1000ba814: fcsel     s2, s5, s2, mi
    1000ba818: fcmp      s1, s2
    1000ba81c: fcsel     s1, s1, s2, mi
    1000ba820: fadd      s1, s1, s3
    1000ba824: str       s1, [x17], #T
    1000ba828: mov.16b   v2, v5
    1000ba82c: subs      x15, x15, #T
    1000ba830: b.ne      T

