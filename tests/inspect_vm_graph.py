"""Inspect Checker DOT dependency graphs. Does not execute VM or measure latency."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def inspect(path):
    text = path.read_text(encoding="utf-8")
    nodes, edges, transfers = {}, defaultdict(set), []
    memory = re.compile(r"(src|dst)\{rank=(\d+), type=(\w+),offset=0x([0-9a-f]+),len=0x([0-9a-f]+)\}")
    for line in text.splitlines():
        match = re.match(r'\s*(v3_\d+) \[label="(.*?)", pos=', line)
        if match:
            node, label = match.groups()
            nodes[node] = label
            if "type=TRANS_MEM" in label:
                row = {"id": node}
                for side, rank, kind, offset, length in memory.findall(label):
                    row[side] = dict(rank=int(rank), kind=kind, offset=int(offset, 16), bytes=int(length, 16))
                assert "src" in row and "dst" in row, label
                assert row["src"]["bytes"] == row["dst"]["bytes"]
                transfers.append(row)
        elif " -> " in line:
            chain = re.findall(r"v3_\d+", line.split("[")[0])
            for left, right in zip(chain, chain[1:]):
                edges[left].add(right)

    def reaches(left, right):
        pending, seen = [left], set()
        while pending:
            current = pending.pop()
            if current == right:
                return True
            if current not in seen:
                seen.add(current)
                pending.extend(edges[current])
        return False

    # Independent scratch stages and forwarded data must have a real causal
    # dependency; the helper's own output must not wait for its forward.
    checks = []
    stages = [t for t in transfers if t["src"]["kind"] == "INPUT" and t["dst"]["kind"] == "CCL"]
    forwards = [t for t in transfers if t["src"]["kind"] == "CCL" and t["dst"]["kind"] == "OUTPUT"]
    for stage in stages:
        scratch = stage["dst"]
        matches = [t for t in forwards if t["src"] == scratch and reaches(stage["id"], t["id"])]
        assert matches, (path.name, "stage has no dependent forward", stage)
        # The same scratch slot may occur in later chunks; choose the first
        # matching forward in the dependency order.
        forward = next(t for t in matches if not any(
            u != t and reaches(u["id"], t["id"]) for u in matches))
        helper = scratch["rank"]
        own = [t for t in transfers if t["src"]["kind"] == "INPUT"
               and t["src"]["rank"] == stage["src"]["rank"]
               and t["dst"]["kind"] == "OUTPUT" and t["dst"]["rank"] == helper
               and t["dst"]["offset"] <= forward["dst"]["offset"]
               < t["dst"]["offset"] + t["dst"]["bytes"]]
        assert len(own) == 1, (path.name, "ambiguous helper output", own)
        assert not reaches(forward["id"], own[0]["id"]), (path.name, "helper output still waits for forwarding")
        assert not reaches(own[0]["id"], forward["id"]), (path.name, "forwarding waits for helper output")
        checks.append(dict(stage=stage["id"], forward=forward["id"], helperOutput=own[0]["id"],
                           helper=helper, outputIndependentOfForward=True))
    assert len(checks) == len(forwards)
    # Every rank's final output is filled exactly once, without a borrowed
    # prefix needing a second restoration write.
    coverage = defaultdict(list)
    for transfer in transfers:
        if transfer["dst"]["kind"] == "OUTPUT":
            dst = transfer["dst"]
            coverage[dst["rank"]].append((dst["offset"], dst["bytes"]))
    totals = {}
    for rank, intervals in coverage.items():
        end = 0
        for offset, size in sorted(intervals):
            assert offset == end, (path.name, rank, "gap or overlapping output", intervals)
            end += size
        totals[rank] = end
    if totals:
        assert len(set(totals.values())) == 1
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                transferKinds=dict(Counter(t["src"]["kind"] + "->" + t["dst"]["kind"] for t in transfers)),
                networkTransfers=sum(t["src"]["rank"] != t["dst"]["rank"] for t in transfers),
                networkBytes=sum(t["src"]["bytes"] for t in transfers if t["src"]["rank"] != t["dst"]["rank"]),
                outputBytesByRank=totals, scratchDependencies=checks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    graphs = sorted(args.data_dir.glob("TaskGraph_*.dot"))
    assert graphs, "No Checker graphs found"
    report = dict(status="PASS", scope="Executed dependency graphs, not performance measurements",
                  graphs=[inspect(path) for path in graphs])
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(status="PASS", graphs=len(graphs),
                         relayEdges=sum(len(g["scratchDependencies"]) for g in report["graphs"]))))
