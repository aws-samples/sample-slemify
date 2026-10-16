#!/usr/bin/env python3
"""Price-performance benchmark for one encoder model served on CPU.

Sends the held-out messages to the model's chat endpoint from N concurrent
clients (closed loop: each client sends the next message when the last one
returns), at several concurrency levels. Before and after each level it reads
the model pod's own /stats (process CPU seconds and requests served), so the
cost is priced on the CPU the model actually used, not on what it requested:

  CPU ms per request = CPU seconds used / requests served
  $ per 1M requests  = CPU seconds per request x node $/hour / node vCPUs / 3600 x 1e6

Runs in-cluster (scripts/run-bench-incluster.sh), next to the model, so a
port-forward is not part of what is measured.
"""
import argparse
import json
import statistics
import threading
import time

import httpx


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))] if xs else 0.0


def stats(client, base):
    return client.get(f"{base}/stats", timeout=10).json()


def body(q):
    return {"model": "model", "messages": [{"role": "user", "content": q}],
            "max_tokens": 32, "temperature": 0.1}


def run_level(base, queries, concurrency, duration):
    lat, errors, lock, nxt = [], [0], threading.Lock(), [0]
    stop_at = time.perf_counter() + duration

    def worker():
        with httpx.Client(timeout=30) as c:
            while time.perf_counter() < stop_at:
                with lock:
                    q = queries[nxt[0] % len(queries)]
                    nxt[0] += 1
                t0 = time.perf_counter()
                try:
                    ok = c.post(f"{base}/v1/chat/completions", json=body(q)).status_code == 200
                except Exception:
                    ok = False
                ms = (time.perf_counter() - t0) * 1000
                with lock:
                    if ok:
                        lat.append(ms)
                    else:
                        errors[0] += 1

    with httpx.Client() as c:
        s0 = stats(c, base)
        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker) for _ in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.perf_counter() - t0
        s1 = stats(c, base)
    served = max(1, s1["requests"] - s0["requests"])
    cpu_s = s1["cpu_seconds"] - s0["cpu_seconds"]
    return {"concurrency": concurrency, "requests": len(lat), "errors": errors[0],
            "rps": len(lat) / wall, "p50_ms": pct(lat, 0.5), "p95_ms": pct(lat, 0.95),
            "cores_busy": cpu_s / wall, "cpu_ms_per_req": cpu_s * 1000 / served,
            "mean_ms": statistics.fmean(lat) if lat else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="model base URL (no path)")
    ap.add_argument("--name", default="model")
    ap.add_argument("--queries", required=True, help="JSON list of messages")
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--duration", type=float, default=10)
    ap.add_argument("--instance-type", default="")
    ap.add_argument("--vcpus", type=float, default=0)
    ap.add_argument("--usd-per-hour", type=float, default=0)
    ap.add_argument("--cpu-request", type=float, default=0, help="the pod's CPU request")
    args = ap.parse_args()

    queries = json.load(open(args.queries))
    with httpx.Client() as c:
        info = stats(c, args.url)
        for q in queries[:20]:  # warm the inference path
            c.post(f"{args.url}/v1/chat/completions", json=body(q), timeout=30)

    usd_per_vcpu_s = (args.usd_per_hour / args.vcpus / 3600) if args.usd_per_hour and args.vcpus else 0
    print(f"Model: {args.name}  precision {info.get('precision') or '?'}, "
          f"{info.get('intra_op_threads') or 'default'} inference threads, CPU request {args.cpu_request:g}")
    if usd_per_vcpu_s:
        print(f"Node:  {args.instance_type} ({args.vcpus:g} vCPUs), on-demand ${args.usd_per_hour:.4f}/hour "
              f"= ${args.usd_per_hour / args.vcpus:.4f} per vCPU-hour")
    print(f"{len(queries)} held-out messages, {args.duration:g} s per level\n")
    print(f"  {'concurrent':>10} {'req/s':>7} {'p50 ms':>7} {'p95 ms':>7} {'cores busy':>10} "
          f"{'CPU ms/req':>10} {'$ per 1M':>9}")
    rows = []
    for lvl in [int(x) for x in args.levels.split(",")]:
        r = run_level(args.url, queries, lvl, args.duration)
        r["usd_per_1m"] = r["cpu_ms_per_req"] / 1000 * usd_per_vcpu_s * 1e6 if usd_per_vcpu_s else None
        rows.append(r)
        usd = f"${r['usd_per_1m']:.2f}" if r["usd_per_1m"] is not None else "n/a"
        print(f"  {lvl:>10} {r['rps']:>7.1f} {r['p50_ms']:>7.1f} {r['p95_ms']:>7.1f} "
              f"{r['cores_busy']:>10.2f} {r['cpu_ms_per_req']:>10.1f} {usd:>9}", flush=True)

    cpu_ms = statistics.median(r["cpu_ms_per_req"] for r in rows)
    one = rows[0]
    summary = {"name": args.name, "precision": info.get("precision"),
               "threads": info.get("intra_op_threads"), "instance_type": args.instance_type,
               "usd_per_hour": args.usd_per_hour, "vcpus": args.vcpus,
               "cpu_request": args.cpu_request, "p50_ms_single": one["p50_ms"],
               "p95_ms_single": one["p95_ms"], "cpu_ms_per_req": cpu_ms,
               "usd_per_1m": cpu_ms / 1000 * usd_per_vcpu_s * 1e6 if usd_per_vcpu_s else None,
               "max_rps": max(r["rps"] for r in rows), "levels": rows}
    if args.cpu_request:
        # What one pod sized at its request can sustain, priced on that slice.
        summary["per_pod_per_hour"] = args.cpu_request * 3600 / (cpu_ms / 1000)
        summary["pod_usd_per_hour"] = args.cpu_request * args.usd_per_hour / args.vcpus if args.vcpus else None
    print()
    line = (f"=== Bench [{args.name} {summary['precision']}]: p50 {one['p50_ms']:.0f} ms one at a time, "
            f"{cpu_ms:.0f} CPU ms per request")
    if summary["usd_per_1m"] is not None:
        line += f", ${summary['usd_per_1m']:.2f} per 1M requests"
    print(line + " ===")
    if summary.get("per_pod_per_hour"):
        print(f"    one pod at its {args.cpu_request:g}-vCPU request: about {summary['per_pod_per_hour']:,.0f} "
              f"requests per hour" + (f" for ${summary['pod_usd_per_hour']:.3f}/hour"
                                      if summary.get("pod_usd_per_hour") else ""))
    print("===BENCH_JSON_BEGIN===")
    print(json.dumps(summary))
    print("===BENCH_JSON_END===")


if __name__ == "__main__":
    main()
