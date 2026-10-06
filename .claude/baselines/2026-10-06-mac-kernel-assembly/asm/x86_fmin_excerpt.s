// x86-64-v3 cross-compile (-target x86_64-apple-macos13 -march=x86-64-v3, same FP flags), probes/x86/lanes_x86.cpp.
// shipped std::min cell: one vminpd per min (x86 minpd has the b < a ? b : a semantics); std::fmin cell: vminpd + vcmpunordpd + vblendvpd per min.
// v_fmin lanes inner loop excerpt:
1067-	vandpd	%ymm13, %ymm12, %ymm12
1068-	vminpd	%ymm5, %ymm10, %ymm14
1069:	vcmpunordpd	%ymm5, %ymm5, %ymm5
1070-	vblendvpd	%ymm5, %ymm10, %ymm14, %ymm5
1071-	vminpd	%ymm5, %ymm7, %ymm14
1072:	vcmpunordpd	%ymm5, %ymm5, %ymm5
1073-	vblendvpd	%ymm5, %ymm7, %ymm14, %ymm5
1074-	vaddpd	%ymm5, %ymm12, %ymm7
1075-	vmovupd	%ymm7, -32(%rbx)
1076-	vsubpd	%ymm4, %ymm11, %ymm5
1077-	vminpd	%ymm6, %ymm9, %ymm11
1078:	vcmpunordpd	%ymm6, %ymm6, %ymm6
1079-	vblendvpd	%ymm6, %ymm9, %ymm11, %ymm6
1080-	vminpd	%ymm6, %ymm8, %ymm11
1081:	vcmpunordpd	%ymm6, %ymm6, %ymm6
1082-	vblendvpd	%ymm6, %ymm8, %ymm11, %ymm6
1083-	vandpd	%ymm5, %ymm13, %ymm5
1084-	vaddpd	%ymm6, %ymm5, %ymm8
1085-	vmovupd	%ymm8, (%rbx)
1086-	incq	%rdi
1087-	addq	$64, %rbx

// shipped lanes inner loop excerpt:
1065-	vsubpd	%ymm3, %ymm11, %ymm12
1066-	vbroadcastsd	LCPI1_2(%rip), %ymm13   ## ymm13 = [NaN,NaN,NaN,NaN]
1067-	vandpd	%ymm13, %ymm12, %ymm12
1068:	vminpd	%ymm5, %ymm9, %ymm5
1069:	vminpd	%ymm5, %ymm7, %ymm5
1070-	vaddpd	%ymm5, %ymm12, %ymm7
1071-	vmovupd	%ymm7, -32(%rbx)
1072-	vsubpd	%ymm4, %ymm11, %ymm5
1073-	vandpd	%ymm5, %ymm13, %ymm5
1074:	vminpd	%ymm6, %ymm10, %ymm6
1075:	vminpd	%ymm6, %ymm8, %ymm6
1076-	vaddpd	%ymm5, %ymm6, %ymm8
1077-	vmovupd	%ymm8, (%rbx)
1078-	incq	%rdi
1079-	addq	$64, %rbx
