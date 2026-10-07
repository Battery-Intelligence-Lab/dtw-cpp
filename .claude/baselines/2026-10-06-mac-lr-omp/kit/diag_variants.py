"""Where does the HiGHS-ON head lose 3 % at N < 280? (stdlib only):  uv run --no-project python diag_variants.py build | run
Four HiGHS-ON probes, each = lrcore_probe.cpp + a scratch compile of the head's dtwc/mip/lagrangian_root.cpp with the head tree's flags, linked as dtwc_cl is:
  base      the base probe (no OpenMP in the LR file)
  head      the head probe as the product is (OpenMP flags, the pragma with its `if` clause, serial branch below 280)
  noomp     the head's source compiled WITHOUT the OpenMP flag and definition (the pragma is ignored: base's code in the head's source)
  nopragma  the head's source with the pragma line deleted, WITH the OpenMP flags (the loop plain, in a file compiled with -fopenmp)
`run` interleaves them on the HiGHS-ON root (stage kel), ON noise N=200 and ON line N=240, REPEATS=9."""
import json
import os
import statistics
import subprocess
import sys

from build_kit import BIN, KIT, PROBE_SRC, TREES, commands, compile_line, link_line, run

src, tree = TREES[("head", "on")]
PRAGMA = "#pragma omp parallel for schedule(static) if(N >= kParallelMinN)\n"
STUB = 'extern "C" long long lr_omp_probe_min_n() { return -1; }\nextern "C" int lr_omp_probe_loop2() { return -1; }\n'


def drop_openmp(args):
    out, skip = [], False
    for k, a in enumerate(args):
        if skip:
            skip = False
        elif a == "-DDTWC_HAS_OPENMP":
            continue
        elif (a == "-Xclang" and args[k + 1] == "-fopenmp") or (a == "-isystem" and "libomp" in args[k + 1]):
            skip = True
        else:
            out.append(a)
    return out


def build(name, text, flags_fn):
    lines = commands(tree)
    cc = compile_line(lines, "dtwc/dtwc_cl.cpp")
    probe_o = BIN / f"diag_{name}_probe.o"
    run([*cc, "-c", str(PROBE_SRC), "-o", str(probe_o)], tree)
    stub = BIN / "diag_stub.cpp"
    stub.write_text(STUB)
    stub_o = BIN / "diag_stub.o"
    run([*cc, "-c", str(stub), "-o", str(stub_o)], tree)
    scratch = BIN / f"diag_{name}.cpp"
    scratch.write_text(text)
    lr = flags_fn(compile_line(lines, "lagrangian_root.cpp"))
    lr_o = BIN / f"diag_{name}.o"
    run([*lr, f"-I{src / 'dtwc/mip'}", "-c", str(scratch), "-o", str(lr_o)], tree)
    ld = link_line(lines)
    i = next(k for k, a in enumerate(ld) if a.endswith("dtwc_cl.cpp.o"))
    out = BIN / f"diag_{name}"
    ld = [*ld[:i], str(probe_o), str(stub_o), str(lr_o), "-o", str(out), *ld[i + 1:]]
    j = ld.index("-o", i + 5)
    del ld[j:j + 2]
    run(ld, tree)
    print("built", out)


def do_build():
    text = (src / "dtwc/mip/lagrangian_root.cpp").read_text()
    assert text.count(PRAGMA) == 1
    build("noomp", text, drop_openmp)
    build("nopragma", text.replace(PRAGMA, ""), lambda a: a)
    build("head", text, lambda a: a)


def do_run():
    probes = {"base": BIN / "probe_base_on", "head": BIN / "diag_head", "noomp": BIN / "diag_noomp", "nopragma": BIN / "diag_nopragma"}
    cells = [("noise", 200, 4), ("line", 240, 3)]
    rep = 9
    for regime, n, k in cells:
        t = {v: [] for v in probes}
        for r in range(rep):
            order = list(probes) if r % 2 == 0 else list(reversed(list(probes)))
            for v in order:
                env = {**os.environ, "OMP_NUM_THREADS": "18", "LRPROBE_MIN_MS": "300", "LRPROBE_STAGES": "kel"}
                env.pop("LR_OMP_MIN_N", None)
                out = subprocess.run([str(probes[v]), "-i", str(KIT / "data" / f"{regime}_N{n}.csv"), "--skip-rows", "1", "--skip-cols", "1", "-k", str(k),
                                      "-m", "lrcore"], capture_output=True, text=True, env=env, timeout=3600).stdout
                for line in out.splitlines():
                    j = json.loads(line)
                    if j["stage"] == "root_kelley":
                        t[v].append((j["ms"], j["mu_fnv"]))
        base = statistics.median(x[0] for x in t["base"])
        same = len({x[1] for v in t for x in t[v]}) == 1
        print(f"HiGHS-ON {regime} N={n} k={k}, root ms (median of {rep}, warm), x = base / variant; multiplier hashes all equal: {same}")
        for v in probes:
            m = statistics.median(x[0] for x in t[v])
            print(f"   {v:9} {m:9.2f}   x{base / m:5.3f}")


if __name__ == "__main__":
    {"build": do_build, "run": do_run}[sys.argv[1]]()
