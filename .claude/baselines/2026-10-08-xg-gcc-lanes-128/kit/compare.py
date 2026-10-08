"""The CLI runs of the base and candidate builds (out-base/, out-cand/ from cli_runs.sh), run by run: every exit 0,
labels and medoids identical, and the differences the contraction may make elsewhere: the total cost (each series'
distance to its medoid, summed, from the written distance matrix, whose entries carry 17 significant digits),
every matrix entry and every silhouette (8 digits), each as the largest relative difference. Writes compare.txt;
its last line, "clustering identical: yes|no", is what ../driver.py's report reads.

usage: uv run --no-project python compare.py
"""
import csv
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
BASE, CAND = HERE / "out-base", HERE / "out-cand"


def rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))[1:]  # past the header


def matrix(path):
    with open(path, newline="") as f:
        return [[float(x) for x in r] for r in csv.reader(f)]


def rel(a, b):
    return 0.0 if a == b else abs(a - b) / max(abs(a), abs(b))


def cost(m, labels, medoids):
    medoid_of = {int(c): int(i) for c, i, _ in medoids}
    return sum(m[i][medoid_of[int(c)]] for i, (_, c) in enumerate(labels))


def exit_ok(d, name):
    return (d / f"{name}.stdout").read_text().rstrip().endswith("exit 0")


out, worst_cost, worst_l1, worst_sq, worst_sil = [], (0.0, ""), 0.0, (0.0, ""), 0.0
same_labels = same_medoids = exits = 0
names = sorted(p.name for p in BASE.iterdir() if p.is_dir())
for name in names:
    b, c = BASE / name, CAND / name
    ok = exit_ok(BASE, name) and exit_ok(CAND, name)
    exits += ok
    lb, lc = rows(b / f"{name}_labels.csv"), rows(c / f"{name}_labels.csv")
    mb, mc = rows(b / f"{name}_medoids.csv"), rows(c / f"{name}_medoids.csv")
    same_labels += lb == lc
    same_medoids += mb == mc
    Mb, Mc = matrix(b / f"{name}_distance_matrix.csv"), matrix(c / f"{name}_distance_matrix.csv")
    dm = max(rel(x, y) for rb, rc in zip(Mb, Mc) for x, y in zip(rb, rc))
    cb, cc = cost(Mb, lb, mb), cost(Mc, lc, mc)
    sb, sc = rows(b / f"{name}_silhouettes.csv"), rows(c / f"{name}_silhouettes.csv")
    ds = max(abs(float(x[2]) - float(y[2])) for x, y in zip(sb, sc))
    worst_sil = max(worst_sil, ds)
    if "_l1_" in name:
        worst_l1 = max(worst_l1, dm)
    elif dm > worst_sq[0]:
        worst_sq = (dm, name)
    if rel(cb, cc) > worst_cost[0]:
        worst_cost = (rel(cb, cc), name)
    out.append(f"{name}: exit {'0/0' if ok else 'NOT 0'}; labels {'same' if lb == lc else 'DIFFER'}; medoids "
               f"{'same' if mb == mc else 'DIFFER'} ({' '.join(r[1] for r in mb)}); cost {cb!r} / {cc!r} (rel "
               f"{rel(cb, cc):.3g}); matrix max rel {dm:.3g}; silhouettes max abs {ds:.3g}")
n = len(names)
identical = n > 0 and exits == n and same_labels == n and same_medoids == n
out += ["",
        f"runs compared: {n} (base and candidate); exit 0 in both: {exits}/{n}; labels identical: {same_labels}/{n}; "
        f"medoids identical: {same_medoids}/{n}",
        f"largest relative difference of the total cost: {worst_cost[0]:.3g} ({worst_cost[1] or 'none'})",
        f"largest relative difference of a distance-matrix entry: l1 runs {worst_l1:.3g}; squared_euclidean runs "
        f"{worst_sq[0]:.3g} ({worst_sq[1] or 'none'})",
        f"largest absolute difference of a silhouette (8 digits written): {worst_sil:.3g}",
        f"clustering identical: {'yes' if identical else 'no'}"]
(HERE / "compare.txt").write_text("\n".join(out) + "\n")
print("\n".join(out[-6:]))
