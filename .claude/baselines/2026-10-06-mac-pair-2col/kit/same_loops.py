"""Do a probe's per-pair Standard loops equal a dtwc_cl's, instruction for instruction (registers renamed)? Compares
the multiset of (kernel, type, cost, instructions, renamed digest) of the no-threshold loops (dtwc_cl has no other).
usage: same_loops.py <dtwc_cl> <probe>"""
import collections
import sys

sys.argv, args = sys.argv[:1], sys.argv[1:]
import placement  # noqa: E402  (same directory)


def key_set(binary):
    return collections.Counter((k, n, norm) for k, _, n, _, _, _, norm in placement.loops_of(binary) if '<true' not in k)


a, b = key_set(args[0]), key_set(args[1])
print(f'{args[0]}: {sum(a.values())} loops; {args[1]}: {sum(b.values())} no-threshold loops; '
      f'equal: {sum((a & b).values())}; only in dtwc_cl: {sum((a - b).values())}; only in probe: {sum((b - a).values())}')
