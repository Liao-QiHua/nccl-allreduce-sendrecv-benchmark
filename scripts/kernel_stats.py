#!/usr/bin/env python3
"""GPU Kernel statistics from Nsight Systems exports.

Reads the SQLite database that `nsys profile` / `nsys export` produces and
reports per-kernel durations for the 8 B stage runs.

Why this exists: averaging every kernel sample silently mixes the 20 warmup
iterations in, and a single stalled kernel (hundreds of milliseconds) then
dominates the mean -- the same trap as the Host trace numbers.  So this script
always drops the first `--warmup` samples per kernel and reports the median next
to the mean, and it flags the spread when the tail is heavy.

Usage:
    python3 scripts/kernel_stats.py <dir-or-sqlite>... [options]

Options:
    --warmup N      drop the first N samples per kernel (default 20)
    --format F      md | csv | tree  (default md)
"""

import argparse
import glob
import os
import sqlite3
import statistics
import sys

US_PER_NS = 1e-3


def kernel_rows(path):
    """Yield (kernel_name, duration_ns) for one Nsight SQLite database."""
    con = sqlite3.connect(path)
    try:
        tables = {r[0] for r in con.execute(
            "select name from sqlite_master where type='table'")}
        if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
            raise SystemExit("%s: no CUPTI_ACTIVITY_KIND_KERNEL table" % path)
        if "StringIds" not in tables:
            raise SystemExit("%s: no StringIds table" % path)
        # Column carrying the kernel name differs across nsys versions.
        cols = {r[1] for r in con.execute(
            "pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
        name_col = "demangledName" if "demangledName" in cols else "shortName"
        query = (
            "select s.value as name, (k.end - k.start) as dur "
            "from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on k.%s = s.id "
            "order by k.start" % name_col)
        for name, dur in con.execute(query):
            yield name, dur
    finally:
        con.close()


def summarize(durations, warmup):
    if len(durations) <= warmup:
        # Never silently fall back to including warmup.
        raise SystemExit(
            "kernel has %d samples, cannot drop %d warmup"
            % (len(durations), warmup))
    kept = durations[warmup:]
    ordered = sorted(kept)
    med = statistics.median(kept)
    return {
        "samples": len(kept),
        "mean_us": sum(kept) / len(kept) * US_PER_NS,
        "median_us": med * US_PER_NS,
        "min_us": ordered[0] * US_PER_NS,
        "max_us": ordered[-1] * US_PER_NS,
        # How badly the mean is dragged by the tail: >1.5 means use the median.
        "mean_over_median": (sum(kept) / len(kept) / med) if med else 0.0,
    }


def expand(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out.extend(sorted(glob.glob(os.path.join(p, "**", "*.sqlite"),
                                        recursive=True)))
        else:
            out.append(p)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-kernel GPU duration statistics from Nsight SQLite.")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--format", choices=("md", "csv", "tree"),
                        default="md")
    args = parser.parse_args(argv)

    files = expand(args.paths)
    if not files:
        raise SystemExit("no .sqlite files found")

    rows = []
    for path in files:
        per_kernel = {}
        for name, dur in kernel_rows(path):
            per_kernel.setdefault(name, []).append(dur)
        for name, durs in sorted(per_kernel.items()):
            st = summarize(durs, args.warmup)
            rows.append(dict(profile=os.path.basename(path), kernel=name, **st))

    if args.format == "csv":
        fields = ["profile", "kernel", "samples", "mean_us", "median_us",
                  "min_us", "max_us", "mean_over_median"]
        print(",".join(fields))
        for r in rows:
            print(",".join(str(r[k]) for k in fields))
        return 0

    if args.format == "tree":
        for r in rows:
            print("%s\n  %s n=%d mean=%.1f us median=%.1f us min=%.1f max=%.1f"
                  % (r["profile"], r["kernel"], r["samples"], r["mean_us"],
                     r["median_us"], r["min_us"], r["max_us"]))
        return 0

    print("| profile | kernel | samples | mean (us) | median (us) | min (us) | max (us) |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        note = ""
        if r["mean_over_median"] >= 1.5:
            note = " (mean inflated by tail)"
        print("| %s | `%s`%s | %d | %.3f | %.3f | %.3f | %.3f |" % (
            r["profile"], r["kernel"], note, r["samples"],
            r["mean_us"], r["median_us"], r["min_us"], r["max_us"]))
    if any(r["mean_over_median"] >= 1.5 for r in rows):
        print()
        print("注：标 (mean inflated by tail) 的行，均值明显高于中位数，"
              "说明存在长尾/停顿样本；小包性能请以中位数为准。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
