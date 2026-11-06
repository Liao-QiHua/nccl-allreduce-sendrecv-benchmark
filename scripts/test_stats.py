#!/usr/bin/env python3
"""Self-test for scripts/stage_stats.py and scripts/kernel_stats.py.

Both scripts exist because ad-hoc statistics went wrong: nanoseconds reported as
microseconds, warmup iterations left in the average, and inclusive intervals
summed as if they nested.  Each of those gets an explicit assertion here.
"""
import csv
import os
import sqlite3
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "stage_stats.py")
KERNEL_SCRIPT = os.path.join(HERE, "kernel_stats.py")

NS = 1
FAIL = []


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print(("  ok   " if ok else "  FAIL ") + "%s: got %.6f want %.6f" % (name, got, want))
    if not ok:
        FAIL.append(name)


def emit(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["event", "rank", "pid", "tid", "bytes",
                    "start_ns", "end_ns", "duration_ns"])
        for r in rows:
            w.writerow([r["event"], r.get("rank", 0), r.get("pid", 100),
                        r.get("tid", 200), 0, r["s"], r["e"], r["e"] - r["s"]])


def allreduce_rows(n_windows, warmup_anchor_us, anchor_us, child_us):
    """Each window: ncclAllReduce > ncclEnqueueCheck, with a known geometry."""
    rows = []
    t = 1000
    for i in range(n_windows):
        a = warmup_anchor_us if i < 20 else anchor_us
        s = t
        e = s + int(a * 1000)
        rows.append({"event": "ncclAllReduce", "s": s, "e": e})
        # child occupies the first `child_us`, exclusive = a - child
        rows.append({"event": "ncclEnqueueCheck",
                     "s": s + 100, "e": s + 100 + int(child_us * 1000)})
        t = e + 50000
    return rows


def union_rows(n_windows):
    """Parent with two overlapping children: union must beat naive sum."""
    rows = []
    t = 1000
    for _ in range(n_windows):
        s = t
        e = s + 10_000  # 10 us parent
        rows.append({"event": "ncclAllReduce", "s": s, "e": e})
        rows.append({"event": "childA", "s": s + 1000, "e": s + 6000})  # 5 us
        rows.append({"event": "childB", "s": s + 4000, "e": s + 8000})  # 4 us
        t = e + 1000
    return rows


def run(tmp, extra=None):
    cmd = [sys.executable, SCRIPT, tmp, "--percentiles"]
    if extra:
        cmd.extend(extra)
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr, file=sys.stderr)
        raise SystemExit("script failed")
    return out.stdout


def parse_tree(text, name):
    for line in text.splitlines():
        if line.strip().startswith(name + "()"):
            body = line.split("[", 1)[1].rstrip("]")
            kv = {}
            for part in body.split(", "):
                if "=" in part:
                    k, v = part.split("=", 1)
                    kv[k.strip()] = v
            return kv
    return None


def test_warmup_and_units(tmp):
    print("test: warmup exclusion + ns->us")
    emit(os.path.join(tmp, "host-Simple.csv.4242"),
         allreduce_rows(120, warmup_anchor_us=5000.0, anchor_us=10.0,
                        child_us=2.0))
    out = run(tmp, ["--format", "tree"])
    kv = parse_tree(out, "ncclAllReduce")
    # 20 warmup windows at 5000 us must be gone; measured anchor is 10 us.
    check("mean inclusive", float(kv["mean inclusive"].split()[0]), 10.0, 1e-3)
    check("mean exclusive", float(kv["mean exclusive"].split()[0]), 8.0, 1e-3)
    kv = parse_tree(out, "ncclEnqueueCheck")
    check("child inclusive", float(kv["mean inclusive"].split()[0]), 2.0, 1e-3)


def test_union_exclusive(tmp):
    print("test: exclusive uses interval union, not a sum")
    emit(os.path.join(tmp, "host-LL.csv.5353"), union_rows(120))
    out = run(tmp, ["--format", "tree", "--warmup", "20"])
    kv = parse_tree(out, "ncclAllReduce")
    # 10 us parent, children cover [1,6)+[4,8) = [1,8) = 7 us -> exclusive 3 us
    check("mean exclusive", float(kv["mean exclusive"].split()[0]), 3.0, 1e-3)
    check("mean inclusive", float(kv["mean inclusive"].split()[0]), 10.0, 1e-3)


def test_na_marker(tmp):
    print("test: unexecuted conditional node is N/A, not 0")
    emit(os.path.join(tmp, "host-Simple.csv.6464"),
         allreduce_rows(120, 10.0, 10.0, 2.0))
    out = run(tmp)
    ok = "ncclP2PPreconnectFunc() [N/A" in out
    print(("  ok   " if ok else "  FAIL ") + "N/A marker present")
    if not ok:
        FAIL.append("N/A marker")


def test_sendrecv_window(tmp):
    print("test: sendrecv window closes on ncclGroupEnd")
    rows = []
    t = 1000
    for _ in range(120):
        s = t
        rows.append({"event": "ncclGroupStart", "s": s, "e": s + 30})
        rows.append({"event": "ncclSend", "s": s + 200, "e": s + 2500})
        rows.append({"event": "ncclRecv", "s": s + 2600, "e": s + 5000})
        rows.append({"event": "ncclGroupEnd", "s": s + 5200, "e": s + 11000})
        t = s + 20000
    emit(os.path.join(tmp, "host-Simple.csv.7373"), rows)
    out = run(tmp)
    kv = parse_tree(out, "ncclGroupEnd")
    check("groupEnd inclusive", float(kv["mean inclusive"].split()[0]), 5.8, 1e-3)
    kv = parse_tree(out, "ncclSend")
    check("send inclusive", float(kv["mean inclusive"].split()[0]), 2.3, 1e-3)


def test_short_trace_fails(tmp):
    print("test: too few windows is an error, not a silent mean")
    emit(os.path.join(tmp, "host-Simple.csv.8484"),
         allreduce_rows(50, 10.0, 10.0, 2.0))
    r = subprocess.run([sys.executable, SCRIPT, tmp],
                       capture_output=True, text=True)
    ok = r.returncode != 0 and "operation windows" in (r.stderr + r.stdout)
    print(("  ok   " if ok else "  FAIL ") + "rejects short trace")
    if not ok:
        FAIL.append("short trace")


def build_nsys_db(path, warmup_ns, steady_ns, tail_ns=None, tail_at=50):
    if tail_ns is None:
        tail_ns = steady_ns
    con = sqlite3.connect(path)
    con.execute("create table StringIds(id integer primary key, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL("
                "start integer, end integer, demangledName integer, "
                "shortName integer)")
    con.execute("insert into StringIds values(1,'kcclKernel(A)')")
    t = 0
    for i in range(120):
        d = warmup_ns if i < 20 else (tail_ns if i == tail_at else steady_ns)
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,1,1)",
                    (t, t + d))
        t += d + 1000
    con.commit()
    con.close()


def test_kernel_warmup(tmp):
    print("test: kernel_stats drops warmup and converts ns->us")
    db = os.path.join(tmp, "8B-Simple-rank0.sqlite")
    build_nsys_db(db, warmup_ns=400_000_000, steady_ns=10_964)
    out = subprocess.run([sys.executable, KERNEL_SCRIPT, db, "--format", "csv"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        raise SystemExit("kernel_stats failed")
    row = out.stdout.strip().splitlines()[1].split(",")
    samples = int(row[2])
    mean_us = float(row[3])
    check("samples", samples, 100)
    # 400 ms warmup samples must not leak into the mean.
    check("mean us", mean_us, 10.964, 1e-3)


def test_kernel_tail_warning(tmp):
    print("test: kernel_stats flags a tail-inflated mean")
    db = os.path.join(tmp, "8B-LL-rank1.sqlite")
    build_nsys_db(db, warmup_ns=1_000, steady_ns=4_032, tail_ns=272_160_000)
    out = subprocess.run([sys.executable, KERNEL_SCRIPT, db],
                         capture_output=True, text=True)
    ok = out.returncode == 0 and "mean inflated by tail" in out.stdout
    print(("  ok   " if ok else "  FAIL ") + "tail flag present")
    if not ok:
        FAIL.append("tail flag")


def test_kernel_short_fails(tmp):
    print("test: kernel_stats refuses to average warmup away when impossible")
    db = os.path.join(tmp, "8B-short.sqlite")
    con = sqlite3.connect(db)
    con.execute("create table StringIds(id integer primary key, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL("
                "start integer, end integer, demangledName integer, "
                "shortName integer)")
    con.execute("insert into StringIds values(1,'k(A)')")
    con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values(0,1000,1,1)")
    con.commit()
    con.close()
    r = subprocess.run([sys.executable, KERNEL_SCRIPT, db],
                       capture_output=True, text=True)
    ok = r.returncode != 0 and "warmup" in (r.stderr + r.stdout)
    print(("  ok   " if ok else "  FAIL ") + "rejects short kernel series")
    if not ok:
        FAIL.append("short kernel series")


def test_proto_from_real_filenames(tmp):
    print("test: protocol parsed from real trace filenames")
    for fname, want in (("host-Simple.csv.1315783", "Simple"),
                        ("host-LL.csv.1315856", "LL")):
        path = os.path.join(tmp, fname)
        emit(path, allreduce_rows(120, 10.0, 10.0, 2.0))
        out = run(tmp)
        ok = out.splitlines()[0].startswith(want + " / rank")
        print(("  ok   " if ok else "  FAIL ") + "%s -> %s" % (fname, want))
        if not ok:
            FAIL.append("proto " + fname)
        os.remove(path)


def test_no_duplicate_children(tmp):
    print("test: each node appears once per path (tree state is reset)")
    emit(os.path.join(tmp, "host-Simple.csv.9191"),
         allreduce_rows(120, 10.0, 10.0, 2.0))
    out = run(tmp)
    body = [ln for ln in out.splitlines() if "()" in ln]
    names = [ln.split("()")[0].strip() for ln in body]
    dup = {n for n in names if names.count(n) > 1}
    ok = not dup
    print(("  ok   " if ok else "  FAIL ") + "no duplicates %s" % (dup or ""))
    if not ok:
        FAIL.append("duplicate children")


def test_outlier_warning_threshold(tmp):
    print("test: outlier warning uses the configured ratio, not 1.0x")
    rows = []
    t = 1000
    for i in range(120):
        # one measured window is 6x the median
        a = 60.0 if i == 60 else 10.0
        s = t
        e = s + int(a * 1000)
        rows.append({"event": "ncclAllReduce", "s": s, "e": e})
        rows.append({"event": "ncclEnqueueCheck",
                     "s": s + 100, "e": s + 100 + 2000})
        t = e + 50000
    emit(os.path.join(tmp, "host-Simple.csv.9292"), rows)
    out = run(tmp)
    ok = ("1 个窗口" in out) and ("最大 6.0 倍" in out)
    print(("  ok   " if ok else "  FAIL ") + "warning reports 1 window at 6.0x")
    if not ok:
        FAIL.append("outlier warning")


def main():
    tests = [test_warmup_and_units, test_union_exclusive, test_na_marker,
             test_sendrecv_window, test_short_trace_fails,
             test_proto_from_real_filenames, test_no_duplicate_children,
             test_outlier_warning_threshold,
             test_kernel_warmup, test_kernel_tail_warning,
             test_kernel_short_fails]
    for t in tests:
        with tempfile.TemporaryDirectory() as tmp:
            t(tmp)
    print()
    if FAIL:
        print("FAILED: %s" % ", ".join(FAIL))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
