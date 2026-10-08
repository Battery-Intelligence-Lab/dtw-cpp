"""Per-process hardware counters from an xperf dumper text of a trace taken with
`xperf -on PROC_THREAD+LOADER+CSWITCH -Pmc <counters> CSWITCH strict`.

Each Pmc row holds the CPU's running counter values at a context switch and the thread switched in; the CSwitch row
that follows it (same timestamp) names that CPU. The difference between two consecutive readings on one CPU belongs
to the thread that ran between them. Usage:
    uv run --no-project python etw_pmc.py <dump.txt> [process-name-substring ...]
Prints the counters summed per process (the named ones, or the ten with the most cycles), with IPC and LLC rates.
"""
import re
import sys
from collections import defaultdict

PMC = re.compile(r"^\s*Pmc,\s*(\d+),\s*(\d+),\s*(.+)$")
CSW = re.compile(r"^\s*CSwitch,\s*(\d+),\s*(.+?) \(\s*(\d+)\),\s*(\d+),")
HEAD = re.compile(r"^\s*Pmc,\s*TimeStamp,\s*ThreadID,\s*(.+)$")


def main(path, wanted):
    names = None
    cpu_last = {}                      # cpu -> (values, tid running since then)
    tid_proc = {}                      # tid -> "name (pid)"
    per_tid = defaultdict(lambda: None)
    pending = None                     # (timestamp, tid, values) awaiting its CSwitch's CPU
    negatives = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if names is None:
                m = HEAD.match(line)
                if m:
                    names = [c.strip() for c in m.group(1).split(",")]
                continue
            m = PMC.match(line)
            if m:
                pending = (int(m.group(1)), int(m.group(2)), [int(v) for v in m.group(3).split(",")])
                continue
            if not line.lstrip().startswith("CSwitch,"):
                continue
            m = CSW.match(line)
            fields = [x.strip() for x in line.split(",")]
            if not m:
                continue
            ts, new_name, new_pid, new_tid = int(m.group(1)), m.group(2).strip(), m.group(3), int(m.group(4))
            tid_proc[new_tid] = f"{new_name} ({new_pid})"
            # Old process and TID, then the CPU column (index 16 counting the event name as 0).
            cpu = int(fields[16])
            if pending is None or pending[0] != ts or pending[1] != new_tid:
                pending = None
                continue
            values = pending[2]
            pending = None
            if cpu in cpu_last:
                prev, ran = cpu_last[cpu]
                delta = [a - b for a, b in zip(values, prev)]
                if min(delta) < 0:
                    negatives += 1
                else:
                    acc = per_tid[ran]
                    per_tid[ran] = delta if acc is None else [a + b for a, b in zip(acc, delta)]
            cpu_last[cpu] = (values, new_tid)
    per_proc = defaultdict(lambda: [0] * len(names))
    for tid, acc in per_tid.items():
        if acc is None:
            continue
        p = tid_proc.get(tid, f"? (tid {tid})")
        per_proc[p] = [a + b for a, b in zip(per_proc[p], acc)]
    rows = [(p, v) for p, v in per_proc.items() if not wanted or any(w.lower() in p.lower() for w in wanted)]
    rows.sort(key=lambda r: -r[1][names.index("UnhaltedCoreCycles")] if "UnhaltedCoreCycles" in names else 0)
    print("counters:", ", ".join(names), f"(negative deltas skipped: {negatives})")
    for p, v in rows[:10]:
        d = dict(zip(names, v))
        ipc = d.get("InstructionRetired", 0) / max(d.get("UnhaltedCoreCycles", 1), 1)
        llc_mpki = 1000 * d.get("LLCMisses", 0) / max(d.get("InstructionRetired", 1), 1)
        llc_ratio = d.get("LLCMisses", 0) / max(d.get("LLCReference", 1), 1)
        print(f"{p:<40} " + "  ".join(f"{k}={d[k]:,}" for k in names)
              + f"  IPC={ipc:.3f}  LLC-MPKI={llc_mpki:.4f}  LLC-miss/ref={llc_ratio:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
