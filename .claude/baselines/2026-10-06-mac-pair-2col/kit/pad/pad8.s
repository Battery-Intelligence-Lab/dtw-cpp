// 8 bytes, ordered first in __text by kit.order (placement p2: every function after it 4 bytes later than in p0)
	.text
	.globl	_kit_pad
	.p2align	2
_kit_pad:
	nop
	ret
