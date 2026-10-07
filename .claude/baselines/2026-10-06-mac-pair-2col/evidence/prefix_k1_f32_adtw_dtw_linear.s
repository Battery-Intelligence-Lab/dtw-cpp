# pprobe_k1_p0 built from the pre-fix k1 header (no one-row guard): f32 ADTW detail::dtw_linear<false>, objdump -d
0000000100012328 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_>:
100012328:     	mov	w8, #0x7f7fffff         ; =2139095039
10001232c:     	fmov	s0, w8
100012330:     	cbz	x0, 0x1000125f8 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2d0>
100012334:     	stp	x26, x25, [sp, #-0x50]!
100012338:     	stp	x24, x23, [sp, #0x10]
10001233c:     	stp	x22, x21, [sp, #0x20]
100012340:     	stp	x20, x19, [sp, #0x30]
100012344:     	stp	x29, x30, [sp, #0x40]
100012348:     	add	x29, sp, #0x40
10001234c:     	mov	x22, x1
100012350:     	cbz	x1, 0x1000125e4 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2bc>
100012354:     	mov	x19, x3
100012358:     	mov	x21, x2
10001235c:     	mov	x20, x0
100012360:     	adrp	x23, 0x100024000 <__ZZN4dtwc4core16dtw_kernel_lanesIdZNS0_20resolve_dtw_block_fnIdEENSt3__18functionIFvNS3_4spanIKT_Lm18446744073709551615EEENS5_IKS8_Lm18446744073709551615EEENS5_IdLm18446744073709551615EEEEEERKNS0_14DistanceConfigEEUlddE_NS0_9LanesCellEEENS3_5arrayIS6_X9dtw_lanesIS6_EEEEPS7_PKSL_miT0_T1_E5y_buf>
100012364:     	add	x23, x23, #0x8b8
100012368:     	ldr	x25, [x23]
10001236c:     	mov	x0, x23
100012370:     	blr	x25
100012374:     	ldrb	w8, [x0]
100012378:     	adrp	x0, 0x100024000 <__ZZN4dtwc4core16dtw_kernel_lanesIdZNS0_20resolve_dtw_block_fnIdEENSt3__18functionIFvNS3_4spanIKT_Lm18446744073709551615EEENS5_IKS8_Lm18446744073709551615EEENS5_IdLm18446744073709551615EEEEEERKNS0_14DistanceConfigEEUlddE_NS0_9LanesCellEEENS3_5arrayIS6_X9dtw_lanesIS6_EEEEPS7_PKSL_miT0_T1_E5y_buf>
10001237c:     	add	x0, x0, #0x8a0
100012380:     	tbz	w8, #0x0, 0x1000125fc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2d4>
100012384:     	ldr	x8, [x0]
100012388:     	blr	x8
10001238c:     	mov	x23, x0
100012390:     	mov	x1, x20
100012394:     	bl	0x100010538 <__ZNSt3__16vectorIfNS_9allocatorIfEEE6resizeEm>
100012398:     	ldr	x8, [x23]
10001239c:     	ldr	s0, [x21]
1000123a0:     	ldr	s1, [x19]
1000123a4:     	fabd	s0, s0, s1
1000123a8:     	str	s0, [x8]
1000123ac:     	cmp	x20, #0x2
1000123b0:     	b.lo	0x1000124dc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x1b4>
1000123b4:     	ldr	s0, [x8]
1000123b8:     	mov	w9, #0x1                ; =1
1000123bc:     	fmov	s1, #0.50000000
1000123c0:     	mov	w10, #0x7f7fffff        ; =2139095039
1000123c4:     	fmov	s2, w10
1000123c8:     	ldr	s3, [x21, x9, lsl #2]
1000123cc:     	ldr	s4, [x19]
1000123d0:     	fabd	s3, s3, s4
1000123d4:     	fadd	s0, s0, s1
1000123d8:     	fminnm	s0, s0, s2
1000123dc:     	fadd	s0, s3, s0
1000123e0:     	str	s0, [x8, x9, lsl #2]
1000123e4:     	add	x9, x9, #0x1
1000123e8:     	cmp	x20, x9
1000123ec:     	b.ne	0x1000123c8 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0xa0>
1000123f0:     	cmp	x22, #0x2
1000123f4:     	b.ls	0x100012544 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x21c>
1000123f8:     	mov	x9, x8
1000123fc:     	ldr	s1, [x9], #0x4
100012400:     	sub	x10, x20, #0x1
100012404:     	add	x11, x21, #0x4
100012408:     	mov	w13, #0x1               ; =1
10001240c:     	mov	w14, #0x2               ; =2
100012410:     	fmov	s0, #0.50000000
100012414:     	mov	w12, #0x7f7fffff        ; =2139095039
100012418:     	fmov	s2, w12
10001241c:     	mov.16b	v4, v1
100012420:     	ldr	s1, [x21]
100012424:     	ldr	s3, [x19, x13, lsl #2]
100012428:     	fabd	s3, s1, s3
10001242c:     	fadd	s5, s4, s0
100012430:     	fminnm	s5, s5, s2
100012434:     	fadd	s3, s3, s5
100012438:     	ldr	s5, [x19, x14, lsl #2]
10001243c:     	fabd	s1, s1, s5
100012440:     	fadd	s5, s3, s0
100012444:     	fminnm	s5, s5, s2
100012448:     	fadd	s1, s5, s1
10001244c:     	str	s1, [x8]
100012450:     	mov	x12, x9
100012454:     	mov	x15, x11
100012458:     	mov	x16, x10
10001245c:     	mov.16b	v5, v1
100012460:     	ldr	s6, [x15], #0x4
100012464:     	ldr	s7, [x19, x13, lsl #2]
100012468:     	ldr	s16, [x12]
10001246c:     	fabd	s7, s6, s7
100012470:     	fadd	s17, s16, s0
100012474:     	fadd	s18, s3, s0
100012478:     	fcmp	s17, s4
10001247c:     	fcsel	s4, s17, s4, mi
100012480:     	fcmp	s18, s4
100012484:     	fcsel	s4, s18, s4, mi
100012488:     	fadd	s7, s4, s7
10001248c:     	ldr	s4, [x19, x14, lsl #2]
100012490:     	fabd	s4, s6, s4
100012494:     	fadd	s6, s7, s0
100012498:     	fadd	s5, s5, s0
10001249c:     	fcmp	s6, s3
1000124a0:     	fcsel	s3, s6, s3, mi
1000124a4:     	fcmp	s5, s3
1000124a8:     	fcsel	s3, s5, s3, mi
1000124ac:     	fadd	s5, s3, s4
1000124b0:     	str	s5, [x12], #0x4
1000124b4:     	mov.16b	v4, v16
1000124b8:     	mov.16b	v3, v7
1000124bc:     	subs	x16, x16, #0x1
1000124c0:     	b.ne	0x100012460 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x138>
1000124c4:     	add	x12, x13, #0x2
1000124c8:     	add	x14, x13, #0x3
1000124cc:     	mov	x13, x12
1000124d0:     	cmp	x14, x22
1000124d4:     	b.lo	0x10001241c <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0xf4>
1000124d8:     	b	0x100012548 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x220>
1000124dc:     	cmp	x22, #0x3
1000124e0:     	b.lo	0x1000125d0 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2a8>
1000124e4:     	add	x10, x19, #0x8
1000124e8:     	mov	w9, #0x2                ; =2
1000124ec:     	fmov	s1, #0.50000000
1000124f0:     	mov	w11, #0x7f7fffff        ; =2139095039
1000124f4:     	fmov	s2, w11
1000124f8:     	ldr	s3, [x21]
1000124fc:     	ldp	s4, s5, [x10, #-0x4]
100012500:     	fabd	s4, s3, s4
100012504:     	fadd	s0, s0, s1
100012508:     	fminnm	s0, s0, s2
10001250c:     	fabd	s3, s3, s5
100012510:     	fadd	s4, s1, s4
100012514:     	fadd	s0, s0, s4
100012518:     	fminnm	s0, s0, s2
10001251c:     	fadd	s0, s0, s3
100012520:     	str	s0, [x8]
100012524:     	add	x10, x10, #0x8
100012528:     	add	x9, x9, #0x2
10001252c:     	cmp	x9, x22
100012530:     	b.lo	0x1000124f8 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x1d0>
100012534:     	sub	x12, x9, #0x1
100012538:     	cmp	x12, x22
10001253c:     	b.lo	0x100012550 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x228>
100012540:     	b	0x1000125dc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2b4>
100012544:     	mov	w12, #0x1               ; =1
100012548:     	cmp	x12, x22
10001254c:     	b.hs	0x1000125dc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2b4>
100012550:     	ldr	s2, [x8]
100012554:     	ldr	s0, [x21]
100012558:     	ldr	s1, [x19, x12, lsl #2]
10001255c:     	fabd	s1, s0, s1
100012560:     	fmov	s0, #0.50000000
100012564:     	fadd	s3, s2, s0
100012568:     	mov	w9, #0x7f7fffff         ; =2139095039
10001256c:     	fmov	s4, w9
100012570:     	fminnm	s3, s3, s4
100012574:     	fadd	s1, s1, s3
100012578:     	str	s1, [x8]
10001257c:     	cmp	x20, #0x2
100012580:     	b.lo	0x1000125dc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2b4>
100012584:     	sub	x9, x20, #0x1
100012588:     	add	x10, x21, #0x4
10001258c:     	add	x11, x8, #0x4
100012590:     	ldr	s3, [x11]
100012594:     	ldr	s4, [x10], #0x4
100012598:     	ldr	s5, [x19, x12, lsl #2]
10001259c:     	fabd	s4, s4, s5
1000125a0:     	fadd	s5, s3, s0
1000125a4:     	fadd	s1, s1, s0
1000125a8:     	fcmp	s5, s2
1000125ac:     	fcsel	s2, s5, s2, mi
1000125b0:     	fcmp	s1, s2
1000125b4:     	fcsel	s1, s1, s2, mi
1000125b8:     	fadd	s1, s1, s4
1000125bc:     	str	s1, [x11], #0x4
1000125c0:     	mov.16b	v2, v3
1000125c4:     	subs	x9, x9, #0x1
1000125c8:     	b.ne	0x100012590 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x268>
1000125cc:     	b	0x1000125dc <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x2b4>
1000125d0:     	mov	w12, #0x1               ; =1
1000125d4:     	cmp	x12, x22
1000125d8:     	b.lo	0x100012550 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x228>
1000125dc:     	add	x8, x8, x20, lsl #2
1000125e0:     	ldur	s0, [x8, #-0x4]
1000125e4:     	ldp	x29, x30, [sp, #0x40]
1000125e8:     	ldp	x20, x19, [sp, #0x30]
1000125ec:     	ldp	x22, x21, [sp, #0x20]
1000125f0:     	ldp	x24, x23, [sp, #0x10]
1000125f4:     	ldp	x26, x25, [sp], #0x50
1000125f8:     	ret
1000125fc:     	ldr	x8, [x0]
100012600:     	mov	x24, x0
100012604:     	blr	x8
100012608:     	mov	x1, x0
10001260c:     	adrp	x0, 0x100010000 <__Z10check_typeIfEyPKci.omp_outlined+0x38a4>
100012610:     	add	x0, x0, #0x504
100012614:     	adrp	x2, 0x100000000 <_strtoul+0x100000000>
100012618:     	add	x2, x2, #0x0
10001261c:     	bl	0x10001d518 <_strtoul+0x10001d518>
100012620:     	mov	x0, x23
100012624:     	blr	x25
100012628:     	mov	x8, x0
10001262c:     	mov	x0, x24
100012630:     	mov	w9, #0x1                ; =1
100012634:     	strb	w9, [x8]
100012638:     	b	0x100012384 <__ZN4dtwc4core6detail10dtw_linearILb0EfNS0_10SpanL1CostIfEENS0_8ADTWCellIfEEEET0_mmT1_T2_S7_+0x5c>

