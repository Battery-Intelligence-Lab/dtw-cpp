#!/usr/bin/env python3
"""Regenerate the mechanical half of .claude/MAP.md (read-only, stdlib only).

  uv run --no-project python scripts/repo_map.py layers [--max-upward N] [--edges A:B ...]
  uv run --no-project python scripts/repo_map.py symbols <doxygen-xml-dir>

`layers` assigns every file under dtwc/ to a layer of the TARGET model
(.claude/design.md section 3) and reports the include graph against it: totals,
layer x layer edge matrix, upward edges, include cycles, fan-in/fan-out and the
transitive weight of the main headers. `--max-upward N` exits 1 when more than N
upward edges exist (a ratchet: the number may only go down).

`symbols` reads Doxygen XML (GENERATE_XML=YES, EXTRACT_ALL=YES over dtwc/) and
lists classes by member count, every enum with its values, and free functions
per namespace.

The layer table below is the single source until W0 Task 9 lands
cmake/DtwcLayers.cmake; after that, keep the two in step (same ranks).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dtwc"
SRC_EXT = (".hpp", ".cpp", ".h", ".cuh", ".cu", ".mm")
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"')

# A layer may include its own layer and any layer of LOWER rank. `io` and
# `backends` share rank 2 and may not include each other.
RANK = {"base": 0, "core": 1, "io": 2, "backends": 2, "algorithms": 3, "mip": 4,
        "session": 5, "surface": 6, "vendored": -1}
ORDER = ["base", "core", "io", "backends", "algorithms", "mip", "session", "surface", "vendored"]

DIR_LAYER = {
    "base": "base", "types": "base", "enums": "base", "core": "core", "detail": "core", "io": "io",
    "cuda": "backends", "metal": "backends", "mpi": "backends",
    "algorithms": "algorithms", "mip": "mip", "cli": "surface", "extern": "vendored",
}
ROOT_FILE_LAYER = {
    # foundation that lives at dtwc/ root until C-11 moves it to dtwc/base/
    "error.hpp": "base", "settings.hpp": "base", "missing_utils.hpp": "base",
    "parallelisation.hpp": "base", "timing.hpp": "base", "system_memory.cpp": "base",
    "system_memory.hpp": "base",  # X-05: gives system_memory.cpp a base-layer declaration
    "env.hpp": "base", "env.cpp": "base",
    # the DTW wrapper family and the series container are core value/kernels code
    "warping.hpp": "core", "warping_adtw.hpp": "core", "warping_ddtw.hpp": "core",
    "warping_missing.hpp": "core", "warping_missing_arow.hpp": "core",
    "warping_wdtw.hpp": "core", "soft_dtw.hpp": "core", "distance.hpp": "core",
    "Data.hpp": "core",
    "DataLoader.hpp": "io", "fileOperations.hpp": "io",
    "initialisation.hpp": "algorithms", "initialisation.cpp": "algorithms",
    "scores.hpp": "algorithms", "scores.cpp": "algorithms",
    "Problem.hpp": "session", "Problem.cpp": "session", "Problem_IO.cpp": "session",
    "checkpoint.hpp": "session", "checkpoint.cpp": "session",
    "api.hpp": "surface", "api.cpp": "surface", "dtwc_cl.cpp": "surface",
    "main.cpp": "surface", "test_api.hpp": "surface", "dtwc.hpp": "surface",
    "utility.hpp": "surface",  # an umbrella header, only dtwc.hpp includes it
}
FILE_OVERRIDE = {
    "core/matrix_io.hpp": "io",
    "core/gpu_dtw_common.hpp": "backends",
    "core/portable_random.hpp": "base", "core/crc32.hpp": "base",
    "core/sha256.hpp": "base", "core/llfio_include.hpp": "base",
    "detail/tier1_method_resolution.hpp": "surface",
}
PROBES = ["core/dtw_kernel.hpp", "core/dtw_dispatch.hpp", "core/pruned_distance_matrix.hpp",
          "warping.hpp", "distance.hpp", "settings.hpp", "Data.hpp", "Problem.hpp",
          "algorithms/fast_pam.hpp", "DataLoader.hpp", "api.hpp", "test_api.hpp", "dtwc.hpp"]


def layer_of(rel: str) -> str:
    if rel in FILE_OVERRIDE:
        return FILE_OVERRIDE[rel]
    parts = rel.split("/")
    if len(parts) == 1:
        return ROOT_FILE_LAYER.get(parts[0], "UNASSIGNED")
    return DIR_LAYER.get(parts[0], "UNASSIGNED")


def allowed(src: str, dst: str) -> bool:
    if "vendored" in (src, dst) or src == dst:
        return True
    return RANK[dst] < RANK[src]


def scan():
    files = {}
    for dirpath, _, names in os.walk(SRC):
        for name in names:
            if name.endswith(SRC_EXT):
                full = os.path.join(dirpath, name)
                files[os.path.relpath(full, SRC).replace(os.sep, "/")] = full
    loc, edges = {}, defaultdict(set)
    for rel, full in files.items():
        with open(full, encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        loc[rel] = len(lines)
        here = os.path.dirname(rel)
        for line in lines:
            match = INCLUDE.match(line)
            if not match:
                continue
            inc = match.group(1)
            candidates = [os.path.normpath(os.path.join(here, inc)), os.path.normpath(inc)]
            if inc.startswith("dtwc/"):
                candidates.append(os.path.normpath(inc[5:]))
            hit = next((c.replace(os.sep, "/") for c in candidates
                        if c.replace(os.sep, "/") in files), None)
            if hit is None:
                # bare includes resolved through a target include dir (e.g. `#include "mip.hpp"`
                # from Problem.cpp via mip-solvers): accept a unique basename match
                by_name = [f for f in files if f.endswith("/" + inc)]
                if len(by_name) == 1:
                    hit = by_name[0]
            if hit and hit != rel:
                edges[rel].add(hit)
    return files, loc, edges


def cycles(files, edges):
    sys.setrecursionlimit(10000)
    index, low, stack, on_stack, out, counter = {}, {}, [], set(), [], [0]

    def strong(v):
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on_stack.add(v)
        for w in edges.get(v, ()):
            if w not in index:
                strong(w)
                low[v] = min(low[v], low[w])
            elif w in on_stack:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                out.append(sorted(comp))

    for v in sorted(files):
        if v not in index:
            strong(v)
    return out


def cmd_layers(args) -> int:
    files, loc, edges = scan()
    totals = defaultdict(lambda: [0, 0])
    for rel in files:
        totals[layer_of(rel)][0] += 1
        totals[layer_of(rel)][1] += loc[rel]
    print("## Layer totals")
    for layer in ORDER + ["UNASSIGNED"]:
        if layer in totals:
            print(f"{layer:11s} rank={RANK.get(layer, '?')!s:>2s} files={totals[layer][0]:3d} loc={totals[layer][1]:6d}")
    unassigned = sorted(r for r in files if layer_of(r) == "UNASSIGNED")
    if unassigned:
        print("UNASSIGNED (add to the table in this script):", ", ".join(unassigned))

    print("\n## Include edges, layer x layer (row includes column)")
    matrix = defaultdict(int)
    for a, targets in edges.items():
        for b in targets:
            matrix[(layer_of(a), layer_of(b))] += 1
    present = [l for l in ORDER if l in totals]
    print(" " * 11 + "".join(f"{l[:8]:>9s}" for l in present))
    for a in present:
        print(f"{a:11s}" + "".join(f"{matrix[(a, b)]:9d}" for b in present))

    print("\n## Upward edges (not allowed by the target model)")
    upward = [(layer_of(a), layer_of(b), a, b)
              for a, targets in sorted(edges.items()) for b in sorted(targets)
              if not allowed(layer_of(a), layer_of(b))]
    by_target = defaultdict(int)
    for la, lb, a, b in upward:
        by_target[b] += 1
        print(f"{la:10s} -> {lb:10s} {a} -> {b}")
    print(f"UPWARD_EDGES={len(upward)}  by included file: " +
          ", ".join(f"{k}={v}" for k, v in sorted(by_target.items(), key=lambda kv: -kv[1])))

    print("\n## Include cycles (file level)")
    found = cycles(files, edges)
    for comp in found:
        print(f"cycle({len(comp)}): " + ", ".join(comp))
    if not found:
        print("none")

    fan_in = defaultdict(int)
    for targets in edges.values():
        for b in targets:
            fan_in[b] += 1
    print("\n## Fan-in top 15")
    for f, n in sorted(fan_in.items(), key=lambda kv: -kv[1])[:15]:
        print(f"in={n:3d} {layer_of(f):10s} {f} ({loc[f]} loc)")
    print("\n## Fan-out top 10")
    for f, targets in sorted(edges.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f"out={len(targets):3d} {layer_of(f):10s} {f} ({loc[f]} loc)")

    print("\n## Transitive weight of key headers (project files pulled in)")
    for probe in PROBES:
        if probe not in files:
            continue
        seen, todo = set(), [probe]
        while todo:
            for w in edges.get(todo.pop(), ()):
                if w not in seen:
                    seen.add(w)
                    todo.append(w)
        by_layer = defaultdict(int)
        for x in seen:
            by_layer[layer_of(x)] += 1
        print(f"{probe:34s} {len(seen):3d} files {sum(loc[x] for x in seen):6d} loc :: " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_layer.items(), key=lambda kv: RANK.get(kv[0], 9))))

    for pair in args.edges or []:
        src, dst = pair.split(":")
        print(f"\n## Edges {src} -> {dst}")
        for a, targets in sorted(edges.items()):
            for b in sorted(targets):
                if layer_of(a) == src and layer_of(b) == dst:
                    print(f"  {a} -> {b}")

    if args.max_upward is not None and len(upward) > args.max_upward:
        print(f"\nFAIL: {len(upward)} upward edges > allowed {args.max_upward}", file=sys.stderr)
        return 1
    return 0


def cmd_symbols(args) -> int:
    classes, enums, functions = [], {}, defaultdict(int)

    def short(path: str | None) -> str:
        if not path:
            return "?"
        return path.split("/dtwc/", 1)[-1] if "/dtwc/" in path else path

    for path in glob.glob(os.path.join(args.xml_dir, "*.xml")):
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        for compound in root.findall("compounddef"):
            kind, name = compound.get("kind"), compound.findtext("compoundname")
            location = compound.find("location")
            where = short(location.get("file") if location is not None else None)
            if kind in ("class", "struct"):
                count = defaultdict(int)
                for member in compound.iter("memberdef"):
                    visibility = "public" if member.get("prot") == "public" else "private"
                    count[(member.get("kind"), visibility)] += 1
                row = (count[("function", "public")], count[("function", "private")],
                       count[("variable", "public")], count[("variable", "private")])
                classes.append((sum(row), kind, name, where, row))
            for member in compound.iter("memberdef"):
                if member.get("kind") == "enum":
                    loc = member.find("location")
                    key = (member.findtext("qualifiedname") or member.findtext("name"),
                           short(loc.get("file") if loc is not None else None))
                    enums[key] = [v.findtext("name") for v in member.findall("enumvalue")]
                elif kind == "namespace" and member.get("kind") == "function":
                    functions[name] += 1

    classes.sort(reverse=True)
    print(f"## Classes and structs: {len(classes)} (public fn / private fn / public var / private var)")
    for total, kind, name, where, row in classes[:args.top]:
        print(f"{total:4d} {kind:6s} {name:56s} {where:36s} " + "/".join(f"{n:3d}" for n in row))
    print("\n## Enums (the settings vocabulary)")
    for (name, where), values in sorted(enums.items(), key=lambda kv: kv[0][1]):
        print(f"{name:46s} {where:34s} {{{', '.join(values)}}}")
    print("\n## Free functions per namespace")
    for name, n in sorted(functions.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{n:4d} {name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    layers = sub.add_parser("layers", help="include graph against the target layer model")
    layers.add_argument("--max-upward", type=int, default=None, help="fail when more upward edges exist")
    layers.add_argument("--edges", nargs="*", metavar="SRC:DST", help="also list the edges between two layers")
    layers.set_defaults(run=cmd_layers)
    symbols = sub.add_parser("symbols", help="class / enum / function inventory from Doxygen XML")
    symbols.add_argument("xml_dir")
    symbols.add_argument("--top", type=int, default=40)
    symbols.set_defaults(run=cmd_symbols)
    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
