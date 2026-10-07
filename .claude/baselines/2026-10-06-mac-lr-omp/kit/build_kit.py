"""Build the four probes of the lr-omp timing kit (stdlib only):
    uv run --no-project python build_kit.py [base_on base_off head_on head_off]    build them (all four by default)
    uv run --no-project python build_kit.py --if-stale                              build all four only if something they embed changed

  bin/probe_base_on    base b36fad43, HiGHS ON  (the Kelley root)       against base-build/
  bin/probe_base_off   base b36fad43, HiGHS OFF (the subgradient root)  against base-build-nohighs/
  bin/probe_head_on    head, HiGHS ON                                   against the worktree's build/
  bin/probe_head_off   head, HiGHS OFF                                  against the worktree's build-nohighs/

Every probe is lrcore_probe.cpp compiled with dtwc_cl.cpp's own flags and linked as dtwc_cl is linked (both read from
`ninja -t commands dtwc_cl` of its tree), so it calls exactly the LR code that tree's dtwc_cl links.
The two head probes take one change: a scratch copy of the tree's dtwc/mip/lagrangian_root.cpp is compiled with the
tree's own flags for that file, its `kParallelMinN` constant replaced by an environment read (LR_OMP_MIN_N, default = the
product's value when the probe was built), and linked ahead of libdtwc_core.a, so libdtwc_core.a's own copy of that member is
not pulled in. The knob lives in the probe only; the product has the constant. A second knob, LR_OMP_LOOP2=0, keeps the second
loop of evaluate_dual (g[j], N*k work) serial whatever N: the experiment the one constant cannot make (default: both loops as
the product). The scratch object also exports the two values it uses, `lr_omp_probe_min_n()` and `lr_omp_probe_loop2()`, which
the probe prints in its `input` line (`min_n_object`, `loop2_object`; the base probes link a stub that says -1), so a transcript
shows what the object really used, not what the environment asked.
A probe embeds the trees' libdtwc_core.a / libdtwc_cli.a and the head's lagrangian_root.cpp: `bin/.stamp` hashes them (and the probe
source and this file), and --if-stale rebuilds when it no longer matches (run.sh calls it).
"""
import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

KIT = Path(__file__).resolve().parent
SCR = Path("/private/tmp/claude-504/-Users-engs2321-git-dtw-cpp/11a5974a-eee8-48c6-bc97-994bfa954411/scratchpad/lr-omp")
WT = Path(os.environ.get("LR_OMP_WORKTREE", "/Users/engs2321/git/dtw-cpp/.claude/worktrees/agent-a51443cdfe91482c7"))
TREES = {
    ("base", "on"): (SCR / "base-src", SCR / "base-build"),
    ("base", "off"): (SCR / "base-src", SCR / "base-build-nohighs"),
    ("head", "on"): (WT, WT / "build"),
    ("head", "off"): (WT, WT / "build-nohighs"),
}
BIN = KIT / "bin"
PROBE_SRC = KIT / "lrcore_probe.cpp"
KNOB_LINE = re.compile(r"\[\[maybe_unused\]\] constexpr index_t kParallelMinN = (\d+);")
STUB = 'extern "C" long long lr_omp_probe_min_n() { return -1; }\nextern "C" int lr_omp_probe_loop2() { return -1; }\n'
EXPORTS = ('\nextern "C" long long lr_omp_probe_min_n() { return static_cast<long long>(dtwc::mip::kParallelMinN); }\n'
           'extern "C" int lr_omp_probe_loop2() { return LOOP2; }\n')


def commands(tree):
    out = subprocess.run(["ninja", "-C", str(tree), "-t", "commands", "dtwc_cl"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()


def strip_options(args, drop_with_value=("-MT", "-MF", "-o", "-c"), drop_flags=("-MD", "-fcolor-diagnostics")):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in drop_with_value:
            skip = True
        elif a not in drop_flags:
            out.append(a)
    return out


def compile_line(lines, source_name):
    line = next(l for l in lines if f"{source_name}" in l and " -c " in l)
    return strip_options(shlex.split(line))


def link_line(lines):
    parts = shlex.split(lines[-1])
    parts = parts[parts.index("&&") + 1:]
    return parts[:parts.index("&&")]


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED: {' '.join(shlex.quote(c) for c in cmd[:6])} ...\n{r.stderr[-3000:]}")
    if r.stderr.strip():
        print(r.stderr[-1500:])


def build(version, flavor):
    src, tree = TREES[(version, flavor)]
    lines = commands(tree)
    cc = compile_line(lines, "dtwc/dtwc_cl.cpp")
    probe_o = BIN / f"lrcore_probe_{version}_{flavor}.o"
    run([*cc, "-c", str(PROBE_SRC), "-o", str(probe_o)], tree)
    objs = [str(probe_o)]
    if version == "base":
        stub = BIN / "lr_omp_probe_stub.cpp"
        stub.write_text(STUB)
        stub_o = BIN / f"lr_omp_probe_stub_{flavor}.o"
        run([*cc, "-c", str(stub), "-o", str(stub_o)], tree)
        objs.append(str(stub_o))
    else:
        # the tree's compile line for lagrangian_root.cpp, on a copy whose constant reads the environment
        text = (src / "dtwc/mip/lagrangian_root.cpp").read_text()
        m = KNOB_LINE.findall(text)
        if len(m) != 1:
            sys.exit(f"build_kit: expected one `[[maybe_unused]] constexpr index_t kParallelMinN = <n>;` in {src}, found {len(m)}")
        knob = (f"index_t lr_omp_min_n() {{ const char *e = std::getenv(\"LR_OMP_MIN_N\"); "
                f"return e ? static_cast<index_t>(std::atoll(e)) : index_t{{ {m[0]} }}; }}\n"
                f"const index_t kParallelMinN = lr_omp_min_n();\n"
                f"const bool lr_omp_loop2_on = [] {{ const char *e = std::getenv(\"LR_OMP_LOOP2\"); return !e || std::atoi(e) != 0; }}();")
        scratch_text = KNOB_LINE.sub(lambda _: knob, text)
        # a second knob for the experiment the brief's single constant cannot make: loop 2 (g[j], N*k work) forked or serial.
        # Two pragmas (the brief's design): the second takes the knob. One pragma (loop 2 left serial in the product): the knob is moot,
        # the object says loop2_object = 0 and LR_OMP_LOOP2 is ignored.
        needle = "schedule(static) if(N >= kParallelMinN)"
        parts = scratch_text.split(needle)
        if len(parts) == 3:
            scratch_text = parts[0] + needle + parts[1] + "schedule(static) if(N >= kParallelMinN && lr_omp_loop2_on)" + parts[2]
        elif len(parts) == 2:
            scratch_text = scratch_text.replace("const bool lr_omp_loop2_on = [] {", "const bool lr_omp_loop2_on_unused = [] {").replace(
                "const bool lr_omp_loop2_on_unused", "[[maybe_unused]] const bool lr_omp_loop2_on_unused")
        else:
            sys.exit(f"build_kit: expected one or two `{needle}` pragmas in {src}, found {len(parts) - 1}")
        loop2_expr = "lr_omp_loop2_on ? 1 : 0" if len(parts) == 3 else "0"
        scratch = BIN / f"lagrangian_root_knob_{flavor}.cpp"
        scratch.write_text("#include <cstdlib>\n" + scratch_text + EXPORTS.replace("LOOP2", "dtwc::mip::" + loop2_expr if "lr_omp_loop2_on" in loop2_expr else loop2_expr))
        lr = compile_line(lines, "lagrangian_root.cpp")
        if not any("openmp" in a for a in lr):
            sys.exit(f"build_kit: the head tree's lagrangian_root.cpp compile line has no OpenMP flag: {lr}")
        lr_o = BIN / f"lagrangian_root_knob_{flavor}.o"
        run([*lr, f"-I{src / 'dtwc/mip'}", "-c", str(scratch), "-o", str(lr_o)], tree)
        objs.append(str(lr_o))
    ld = link_line(lines)
    i = next(k for k, a in enumerate(ld) if a.endswith("dtwc_cl.cpp.o"))
    out = BIN / f"probe_{version}_{flavor}"
    ld = [*ld[:i], *objs, "-o", str(out), *ld[i + 1:]]
    j = ld.index("-o", i + len(objs) + 2)  # the original `-o bin/dtwc_cl` follows the one just inserted
    del ld[j:j + 2]
    run(ld, tree)
    print(f"built {out}")


def stamp():
    """a hash of everything a probe embeds: the probe source, this file, the head's and base's lagrangian_root.cpp, the four trees' libraries"""
    h = hashlib.sha256()
    for f in (PROBE_SRC, Path(__file__), WT / "dtwc/mip/lagrangian_root.cpp", SCR / "base-src/dtwc/mip/lagrangian_root.cpp"):
        h.update(f.read_bytes())
    for key in sorted(TREES):
        for lib in ("libdtwc_core.a", "libdtwc_cli.a"):
            p = TREES[key][1] / "bin" / lib
            st = p.stat()
            h.update(f"{p}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()


if __name__ == "__main__":
    BIN.mkdir(exist_ok=True)
    args = sys.argv[1:]
    if args == ["--if-stale"]:
        names = [f"probe_{v}_{f}" for v in ("base", "head") for f in ("on", "off")]
        marker = BIN / ".stamp"
        if all((BIN / n).exists() for n in names) and marker.exists() and marker.read_text().strip() == stamp():
            print("probes up to date")
            sys.exit(0)
        print("probes missing or stale: rebuilding all four")
        args = []
    wanted = args or [f"{v}_{f}" for v in ("base", "head") for f in ("on", "off")]
    for w in wanted:
        build(*w.split("_"))
    if len(wanted) == 4:
        (BIN / ".stamp").write_text(stamp() + "\n")
