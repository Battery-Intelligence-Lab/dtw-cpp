.intel_syntax noprefix
vbroadcastsd	ymm0, qword ptr [rsi + 8*rdx]
vmovupd	ymm8, ymmword ptr [r12 - 0x60]
vsubpd	ymm2, ymm0, ymm4
vminpd	ymm14, ymm8, ymm15
vminpd	ymm3, ymm3, ymm14
vbroadcastsd	ymm14, qword ptr [rip + 0x5d421c]
vandpd	ymm2, ymm14, ymm2
vaddpd	ymm3, ymm3, ymm2
vmovupd	ymm15, ymmword ptr [r12 - 0x40]
vminpd	ymm2, ymm15, ymm9
vminpd	ymm1, ymm1, ymm2
vsubpd	ymm2, ymm0, ymm5
vandpd	ymm2, ymm14, ymm2
vaddpd	ymm1, ymm1, ymm2
vmovupd	ymm2, ymmword ptr [r12 - 0x20]
vminpd	ymm9, ymm2, ymm11
vminpd	ymm9, ymm10, ymm9
vsubpd	ymm10, ymm0, ymm6
vandpd	ymm10, ymm10, ymm14
vaddpd	ymm10, ymm9, ymm10
vsubpd	ymm0, ymm0, ymm7
vandpd	ymm0, ymm14, ymm0
vmovupd	ymm9, ymmword ptr [r12]
vminpd	ymm11, ymm9, ymm13
vminpd	ymm11, ymm12, ymm11
vaddpd	ymm12, ymm11, ymm0
vmovupd	ymmword ptr [r12 - 0x60], ymm3
vmovupd	ymmword ptr [r12 - 0x40], ymm1
vmovupd	ymmword ptr [r12 - 0x20], ymm10
vmovupd	ymmword ptr [r12], ymm12
inc	rdx
sub	r12, -0x80
vmovapd	ymm13, ymm9
vmovapd	ymm11, ymm2
vmovapd	ymm9, ymm15
vmovapd	ymm15, ymm8
cmp	rdi, rdx
