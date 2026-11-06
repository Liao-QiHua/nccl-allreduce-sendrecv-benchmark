#!/usr/bin/env python3
"""Stage statistics for NCCL Host Trace CSVs.

Rebuilds per-rank Host call trees for the 8 B stage runs produced by
scripts/run_stage_trace.sh (host-<proto>.csv.<pid>).

Design notes -- these are exactly the points where ad-hoc analysis went wrong
before, so they are enforced here rather than left to the caller:

1. Unit.  The trace stores duration_ns / start_ns / end_ns in nanoseconds.
   Every reported value is converted to microseconds (ns / 1000).
2. Warmup.  The benchmark does 20 warmup + 100 measured iterations.  The first
   `--warmup` operation windows of each (pid, operation) are dropped BEFORE any
   aggregation.  Leaving them in is what produced the "groupLaunch mean=4999 us"
   nonsense earlier: the very first window contains the one-off P2P preconnect
   stall (~0.5 s).
3. Exclusive.  Parent/child inclusive intervals nest, so they must never be
   summed.  Exclusive is `duration - union(direct children on the same TID)`,
   using an interval union, not a sum.
4. Conditional paths.  Functions that never ran are reported as N/A, never 0 us.
5. Outliers are counted and surfaced, not silently averaged away.

Usage:
    python3 scripts/stage_stats.py <trace_dir> [options]

    <trace_dir> holds host-Simple.csv.<pid> / host-LL.csv.<pid> and the
    8B-<proto>.csv end-to-end CSVs written by scripts/run_stage_trace.sh.

Options:
    --warmup N        operation windows to drop (default 20)
    --iters N         measured windows to keep (default 100)
    --format F        tree | csv | md   (default tree)
    --percentiles     also print P50/P90/P99/min/max (default: means only)
    --max-anchor-ratio R   flag windows whose anchor duration exceeds
                           R x the median anchor duration (default 5.0)
"""

import argparse
import csv
import glob
import os
import statistics
import sys
from collections import defaultdict

US_PER_NS = 1e-3

# Function nodes we expect on each path, used to emit "N/A" for absent ones.
EXPECTED_ALLREDUCE = [
    "ncclAllReduce",
    "ncclEnqueueCheck",
    "taskAppend",
    "collTaskAppend",
    "ncclGroupEndInternal",
    "groupLaunch",
    "ncclP2PPreconnectFunc",
    "ncclPrepareTasksAndCollPreconnect",
    "ncclPrepareTasks",
    "ncclPrepareTasksAndCollPreconnectFunc",
    "ncclTasksRegAndEnqueue",
    "doLaunches",
    "ncclLaunchPrepare",
    "scheduleCollTasksToPlan",
    "scheduleP2pTasksToPlan",
    "ncclLaunchKernelBefore_NoUncapturedCuda",
    "ncclLaunchKernel",
    "ncclLaunchKernelAfter_NoCuda",
    "ncclLaunchFinish",
]

EXPECTED_SENDRECV = [
    "ncclGroupStart",
    "ncclSend",
    "ncclRecv",
    "ncclEnqueueCheck",
    "taskAppend",
    "p2pTaskAppend",
    "ncclGroupEnd",
    "ncclGroupEndInternal",
    "groupLaunch",
    "ncclP2PPreconnectFunc",
    "ncclPrepareTasksAndCollPreconnect",
    "ncclPrepareTasks",
    "ncclPrepareTasksAndCollPreconnectFunc",
    "ncclTasksRegAndEnqueue",
    "doLaunches",
    "ncclLaunchPrepare",
    "scheduleCollTasksToPlan",
    "scheduleP2pTasksToPlan",
    "ncclLaunchKernelBefore_NoUncapturedCuda",
    "ncclLaunchKernel",
    "ncclLaunchKernelAfter_NoCuda",
    "ncclLaunchFinish",
]

# Anchor event of one measured operation window.
ANCHORS = {"allreduce": "ncclAllReduce", "sendrecv": "ncclGroupStart"}
# End marker for the sendrecv window (group scope closes on ncclGroupEnd).
WINDOW_END = {"allreduce": "ncclAllReduce", "sendrecv": "ncclGroupEnd"}


class Event(object):
    __slots__ = ("name", "rank", "pid", "tid", "start", "end", "dur",
                 "parent", "kids", "excl")

    def __init__(self, name, rank, pid, tid, start, end):
        self.name = name
        self.rank = rank
        self.pid = pid
        self.tid = tid
        self.start = start
        self.end = end
        self.dur = (end - start) * US_PER_NS
        self.parent = None
        self.kids = []
        self.excl = self.dur


def read_trace(path):
    """Read one host-<proto>.csv.<pid> file. Returns (proto, events)."""
    base = os.path.basename(path)
    proto = "Simple" if ".Simple." in base else "LL" if ".LL." in base else "?"
    events = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"event", "rank", "pid", "tid", "start_ns", "end_ns"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit("%s: missing columns %s" % (path, sorted(missing)))
        for row in reader:
            name = (row.get("event") or "").strip()
            if not name:
                continue
            try:
                start = int(row["start_ns"])
                end = int(row["end_ns"])
            except (TypeError, ValueError):
                continue
            events.append(Event(
                name=name,
                rank=int(row["rank"]),
                pid=int(row["pid"]),
                tid=int(row["tid"]),
                start=start,
                end=end,
            ))
    if not events:
        raise SystemExit("%s: no usable events" % path)
    events.sort(key=lambda e: (e.start, -e.end))
    return proto, events


def operation_windows(events, operation, warmup, iters):
    """Select the measured operation windows for one operation type.

    Returns a list of (start_ns, end_ns) covering exactly `iters` windows with
    the first `warmup` windows removed.
    """
    anchor = ANCHORS[operation]
    ends = [e for e in events if e.name == WINDOW_END[operation]]
    windows = []
    for a in (e for e in events if e.name == anchor):
        if operation == "allreduce":
            windows.append((a.start, a.end, a))
            continue
        # sendrecv: close on the first ncclGroupEnd starting at/after the anchor.
        closing = next((z for z in ends if z.start >= a.start), None)
        if closing is None:
            continue
        windows.append((a.start, closing.end, a))

    total = len(windows)
    if total == 0:
        # A file may legitimately hold only one of the two cases.
        return [], 0
    if total < warmup + iters:
        raise SystemExit(
            "only %d %s operation windows found, need %d (warmup %d + iters %d)"
            % (total, operation, warmup + iters, warmup, iters))
    return windows[warmup:warmup + iters], total


def build_tree(events, start, end, tid):
    """Events belonging to [start, end) on `tid`, nested by time containment.

    Returns (roots, async_count) where async_count counts events in the window
    that live on another TID (async callbacks must not be folded into the
    synchronous stack).
    """
    inside = [e for e in events if e.start >= start and e.start < end]
    async_count = sum(1 for e in inside if e.tid != tid)
    stack = []
    roots = []
    for e in (x for x in inside if x.tid == tid):
        while stack and stack[-1].end <= e.start:
            stack.pop()
        if stack and e.end > stack[-1].end:
            # Overlapping but not nested.  Do not invent a parent: that would
            # misreport the call path.  It stays a sibling; its interval is
            # still excluded from the enclosing scope's exclusive time below.
            e.parent = None
            roots.append(e)
            stack.append(e)
            continue
        if stack:
            e.parent = stack[-1]
            stack[-1].kids.append(e)
        else:
            roots.append(e)
        stack.append(e)
    # Exclusive = own duration minus the union of every interval strictly
    # nested inside it on the same TID.  Union of all descendants equals union
    # of direct children for properly nested scopes, and stays correct even if
    # the trace contains a partial overlap.
    siblings = [x for x in inside if x.tid == tid]
    for e in siblings:
        contained = [o for o in siblings
                     if o is not e and o.start >= e.start and o.end <= e.end]
        e.excl = e.dur - union_length(contained)
    return roots, async_count


def union_length(kids):
    """Length of the union of child intervals, in microseconds."""
    if not kids:
        return 0.0
    spans = sorted((k.start, k.end) for k in kids)
    total = 0
    cur_s, cur_e = spans[0]
    for s, e in spans[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        elif e > cur_e:
            cur_e = e
    total += cur_e - cur_s
    return total * US_PER_NS


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def collect(events, windows, tid, operation):
    """Aggregate per-event-name statistics across all measured windows."""
    incl = defaultdict(list)
    excl = defaultdict(list)
    outliers = 0
    anchor_ratio = 0.0

    anchor_durs = [(w[1] - w[0]) * US_PER_NS for w in windows]
    median_anchor = statistics.median(anchor_durs) if anchor_durs else 0.0

    representative = None
    for start, end, anchor in windows:
        roots, _ = build_tree(events, start, end, tid)
        for root in roots:
            walk(root, incl, excl)
        d = (end - start) * US_PER_NS
        if median_anchor > 0 and d > max(anchor_ratio, 1.0) * median_anchor:
            outliers += 1
            anchor_ratio = max(anchor_ratio, d / median_anchor)
        if representative is None or abs(d - median_anchor) < abs(
                (representative[1] - representative[0]) * US_PER_NS - median_anchor):
            representative = (start, end, anchor)

    if representative is None:
        return None
    roots, async_count = build_tree(events, *representative[:2], tid=tid)
    stats = {}
    for name in set(list(incl) + list(excl)):
        i = incl[name]
        x = excl[name]
        stats[name] = {
            "calls": len(i),
            "mean_incl": sum(i) / len(i),
            "mean_excl": sum(x) / len(x),
            "p50": pct(i, 50),
            "p90": pct(i, 90),
            "p99": pct(i, 99),
            "min": min(i),
            "max": max(i),
        }
    return {
        "roots": roots,
        "stats": stats,
        "async": async_count,
        "outliers": outliers,
        "max_ratio": anchor_ratio,
        "anchor_us": (representative[1] - representative[0]) * US_PER_NS,
        "representative": representative,
        "operation": operation,
        "tid": tid,
    }


def walk(event, incl, excl):
    incl[event.name].append(event.dur)
    excl[event.name].append(max(event.excl, 0.0))
    for kid in event.kids:
        walk(kid, incl, excl)


def fmt_node(name, st, percentiles):
    if percentiles:
        return ("%s() [calls=%d, mean inclusive=%.3f us, mean exclusive=%.3f us, "
                "P50=%.3f, P90=%.3f, P99=%.3f, min=%.3f, max=%.3f]" % (
                    name, st["calls"], st["mean_incl"], st["mean_excl"],
                    st["p50"], st["p90"], st["p99"], st["min"], st["max"]))
    return "%s() [mean inclusive=%.3f us, mean exclusive=%.3f us]" % (
        name, st["mean_incl"], st["mean_excl"])


def render_tree(event, depth, stats, percentiles, out):
    st = stats.get(event.name)
    if st is None:
        out.append("  " * depth + "%s() [N/A (insufficient event attribution)]"
                   % event.name)
    else:
        out.append("  " * depth + fmt_node(event.name, st, percentiles))
    for kid in sorted(event.kids, key=lambda k: k.start):
        render_tree(kid, depth + 1, stats, percentiles, out)


def render_missing(expected, stats, roots, out):
    present = set(stats)
    tree_names = set()

    def walk_names(e):
        tree_names.add(e.name)
        for k in e.kids:
            walk_names(k)

    for r in roots:
        walk_names(r)
    missing = [n for n in expected if n not in present and n not in tree_names]
    if missing:
        out.append("")
        out.append("未执行的节点（N/A，非 0 us）：")
        for n in missing:
            out.append("  %s() [N/A (not executed in window)]" % n)


def report(trace_dir, args):
    files = sorted(glob.glob(os.path.join(trace_dir, "host-*.csv.*")))
    if not files:
        raise SystemExit("no host-*.csv.<pid> files under %s" % trace_dir)

    if args.percentiles:
        pass

    rows = []
    blocks = []
    for path in files:
        proto, events = read_trace(path)
        pid = events[0].pid
        for operation in ("allreduce", "sendrecv"):
            windows, total = operation_windows(
                events, operation, args.warmup, args.iters)
            if total == 0:
                continue
            # Group by TID; the API thread is the one carrying the anchor.
            tids = sorted({e.tid for e in events
                           if e.name == ANCHORS[operation]})
            for tid in tids:
                result = collect(events, windows, tid, operation)
                if result is None:
                    continue
                rank = max((e.rank for e in result["roots"]), default=-1)
                if rank < 0:
                    rank = max((e.rank for e in events if e.rank >= 0),
                               default=-1)
                header = ("%s / rank %s / pid %d / tid %d / %s"
                          % (proto, rank, pid, tid, operation))
                meta = ("windows=%d (of %d total), warmup dropped=%d, "
                        "anchor(median window)=%.3f us"
                        % (args.iters, total, args.warmup,
                           result["anchor_us"]))
                out = [header, meta]
                if result["async"]:
                    out.append("注意: 窗口内有 %d 个事件位于其他 TID（异步回调），"
                               "未计入同步调用栈" % result["async"])
                if result["outliers"]:
                    out.append("警告: %d 个窗口的锚点耗时超过中位数的 %.1f 倍"
                               "（最大 %.1f 倍）；均值已包含这些窗口"
                               % (result["outliers"], args.max_anchor_ratio,
                                  result["max_ratio"]))
                out.append("")
                for root in result["roots"]:
                    body = []
                    render_tree(root, 0, result["stats"], args.percentiles, body)
                    out.extend(body)
                expected = (EXPECTED_ALLREDUCE if operation == "allreduce"
                            else EXPECTED_SENDRECV)
                render_missing(expected, result["stats"], result["roots"], out)
                blocks.append("\n".join(out))

                for name, st in sorted(result["stats"].items()):
                    rows.append({
                        "protocol": proto,
                        "rank": rank,
                        "pid": pid,
                        "tid": tid,
                        "operation": operation,
                        "function": name,
                        "calls": st["calls"],
                        "mean_inclusive_us": st["mean_incl"],
                        "mean_exclusive_us": st["mean_excl"],
                        "p50_us": st["p50"],
                        "p90_us": st["p90"],
                        "p99_us": st["p99"],
                        "min_us": st["min"],
                        "max_us": st["max"],
                    })

    if args.format == "tree":
        print("\n\n".join(blocks))
        return

    fields = ["protocol", "rank", "pid", "tid", "operation", "function", "calls",
              "mean_inclusive_us", "mean_exclusive_us",
              "p50_us", "p90_us", "p99_us", "min_us", "max_us"]
    if args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return

    # markdown
    print("| protocol | rank | operation | function | calls | mean incl (us) | mean excl (us) |")
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print("| %s | %s | %s | `%s` | %d | %.3f | %.3f |" % (
            row["protocol"], row["rank"], row["operation"], row["function"],
            row["calls"], row["mean_inclusive_us"], row["mean_exclusive_us"]))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rebuild NCCL Host stage statistics from Host Trace CSVs.")
    parser.add_argument("trace_dir")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--format", choices=("tree", "csv", "md"),
                        default="tree")
    parser.add_argument("--percentiles", action="store_true",
                        help="also print P50/P90/P99/min/max")
    parser.add_argument("--max-anchor-ratio", type=float, default=5.0,
                        help="warn when an anchor exceeds R x the median")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iters <= 0:
        raise SystemExit("warmup must be >= 0 and iters > 0")
    report(args.trace_dir, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
