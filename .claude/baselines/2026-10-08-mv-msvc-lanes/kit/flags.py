"""Print the compile command of given sources from a build tree's compile_commands.json (read-only).
usage: flags.py <build dir> <source suffix>..."""
import json
import sys

entries = json.load(open(sys.argv[1] + "/compile_commands.json"))
for suffix in sys.argv[2:]:
    hits = [e for e in entries if e["file"].replace("\\", "/").endswith(suffix)]
    print("==", suffix, len(hits), "hit(s)")
    for e in hits:
        print(e.get("command") or " ".join(e.get("arguments", [])))
