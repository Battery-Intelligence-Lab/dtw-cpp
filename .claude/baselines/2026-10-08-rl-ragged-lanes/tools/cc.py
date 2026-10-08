"""Print the compile command of each source whose path ends with one of the given suffixes, from a build tree's
compile_commands.json. usage: cc.py <build dir> <suffix> [suffix ...]"""
import json
import sys

for e in json.load(open(sys.argv[1] + "/compile_commands.json")):
    f = e["file"].replace("\\", "/")
    if f.endswith(tuple(sys.argv[2:])):
        print(f)
        print(e.get("command") or " ".join(e["arguments"]))
        print()
