"""Report how the site's 429 rate limit behaves, from the scraper's logs.

Answers what the limit is keyed on, which decides the fix: a per-worker
limit is beaten by pacing, a per-mirror one by spreading workers across
mirrors, and a global one only by sending less. Reads every rotated log
file (logs/darknet-scraper.log*) and needs nothing beyond the stdlib.
"""

import argparse
import math
import re
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

# Thread name is optional: logs from before the worker threads didn't have it.
_LINE_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}) (\w+) (?:(\S+) )?(darknet_scraper\S*): (.*)$"
)
_SEND_RE = re.compile(r"^(GET|POST) (\S+) \(attempt \d+")
_STATUS_RE = re.compile(r"^(GET|POST) (\S+) -> (\d{3})$")
_LIMITED_RE = re.compile(
    r"^Rate limited \(429\) .*request_delay now ([\d.]+)s"
    r"(?:, host=\S+, retry_after=([^,]+), limit_headers=(.*))?$"
)
_GAVE_UP = "Still rate limited (429) after"
_FIRST_BODY = "First 429 body for "

GAP_BUCKETS = [0, 5, 10, 20, 30, 45, 60, 120, math.inf]


def _log_files(log_dir: Path) -> list[Path]:
    # Oldest first: darknet-scraper.log.5 ... .log.1, then the live .log.
    rotated = sorted(
        log_dir.glob("darknet-scraper.log.*"),
        key=lambda p: int(p.suffix[1:]) if p.suffix[1:].isdigit() else 0,
        reverse=True,
    )
    live = log_dir / "darknet-scraper.log"
    return rotated + ([live] if live.exists() else [])


def _parse(files: list[Path]) -> list[tuple[datetime, str, str]]:
    events = []
    for path in files:
        with open(path, errors="replace") as fh:
            for line in fh:
                m = _LINE_RE.match(line.rstrip("\n"))
                if not m:
                    continue
                ts, _level, thread, _name, msg = m.groups()
                worker = thread if thread and thread.startswith("worker") else "main"
                events.append((datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f"), worker, msg))
    # Stable: lines with the same timestamp keep their file order.
    events.sort(key=lambda e: e[0])
    return events


def _rate(n: int, total: int) -> str:
    return f"{n / total:6.1%}" if total else "     -"


def report(log_dir: Path, window: float) -> None:
    files = _log_files(log_dir)
    if not files:
        print(f"No logs found in {log_dir}")
        return
    events = _parse(files)
    if not events:
        print(f"No parsable log lines in {', '.join(map(str, files))}")
        return

    requests = Counter()  # worker -> responses
    limited = Counter()  # worker -> 429s
    delays = defaultdict(list)  # worker -> request_delay after each 429
    host_requests = Counter()
    host_limited = Counter()
    hour_requests = Counter()
    hour_limited = Counter()
    limited_at = defaultdict(list)  # worker -> 429 timestamps (seconds)
    gap_requests = Counter()  # gap bucket -> responses
    gap_limited = Counter()
    last_send = {}  # worker -> timestamp of its previous request
    pending_gap = {}  # worker -> gap before the request awaiting its status
    retry_after = Counter()
    limit_headers = Counter()
    gave_up = Counter()
    first_bodies = []

    for ts, worker, msg in events:
        if m := _SEND_RE.match(msg):
            prev = last_send.get(worker)
            pending_gap[worker] = (ts - prev).total_seconds() if prev else None
            last_send[worker] = ts
        elif m := _STATUS_RE.match(msg):
            _method, url, status = m.groups()
            is_429 = status == "429"
            host = urlsplit(url).netloc
            hour = ts.strftime("%Y-%m-%d %H:00")
            requests[worker] += 1
            host_requests[host] += 1
            hour_requests[hour] += 1
            if is_429:
                limited[worker] += 1
                host_limited[host] += 1
                hour_limited[hour] += 1
                limited_at[worker].append(ts.timestamp())
            gap = pending_gap.pop(worker, None)
            if gap is not None:
                bucket = next(b for b in GAP_BUCKETS[1:] if gap < b)
                gap_requests[bucket] += 1
                gap_limited[bucket] += is_429
        elif m := _LIMITED_RE.match(msg):
            delays[worker].append(float(m.group(1)))
            if m.group(2) is not None:
                retry_after[m.group(2).strip()] += 1
                limit_headers[m.group(3).strip()] += 1
        elif msg.startswith(_GAVE_UP):
            gave_up[worker] += 1
        elif msg.startswith(_FIRST_BODY) and len(first_bodies) < 3:
            first_bodies.append(f"{worker}: {msg[len(_FIRST_BODY):]}")

    span = (events[-1][0] - events[0][0]).total_seconds()
    total_req = sum(requests.values())
    total_429 = sum(limited.values())
    print(f"logs: {len(files)} file(s), {events[0][0]} -> {events[-1][0]} ({span / 3600:.1f}h)")
    print(f"responses: {total_req}, 429s: {total_429} ({_rate(total_429, total_req).strip()})")

    print("\n1. per worker")
    print(f"  {'worker':10} {'requests':>9} {'429s':>7} {'rate':>7} {'median delay':>13} {'gave up':>8}")
    for w in sorted(requests):
        med = f"{statistics.median(delays[w]):.1f}s" if delays[w] else "-"
        print(
            f"  {w:10} {requests[w]:9} {limited[w]:7} {_rate(limited[w], requests[w])} "
            f"{med:>13} {gave_up[w]:8}"
        )

    print("\n2. per mirror host")
    for host, n in host_requests.most_common():
        print(f"  {host:66} {n:8} req  {host_limited[host]:7} 429  {_rate(host_limited[host], n)}")

    print("\n3. per hour (last 48)")
    for hour in sorted(hour_requests)[-48:]:
        n = hour_requests[hour]
        print(f"  {hour}  {n:6} req  {hour_limited[hour]:5} 429  {_rate(hour_limited[hour], n)}")

    print(f"\n4. cross-worker clustering (another worker's 429 within ±{window:.0f}s)")
    workers = [w for w in limited_at if limited_at[w] and w != "main"]
    # Only the time the workers ran in parallel: older single-process logs
    # would dilute the baseline and make any overlap look significant.
    worker_ts = [ts for ts, w, _ in events if w != "main"]
    worker_span = (worker_ts[-1] - worker_ts[0]).total_seconds() if worker_ts else 0
    if len(workers) < 2 or worker_span <= 0:
        print("  needs 429s from at least 2 workers")
    else:
        hits = 0
        expected = 0.0
        for w in workers:
            others = sorted(t for o in workers if o != w for t in limited_at[o])
            # If 429s were independent, other workers' 429s land in a
            # 2*window slot at random with this probability.
            expected += len(limited_at[w]) * (
                1 - math.exp(-len(others) / worker_span * 2 * window)
            )
            for t in limited_at[w]:
                if bisect_right(others, t + window) > bisect_left(others, t - window):
                    hits += 1
        n = sum(len(limited_at[w]) for w in workers)
        print(f"  observed: {hits}/{n} ({_rate(hits, n).strip()})")
        print(f"  expected if independent: {expected / n:.1%}")
        print(
            "  -> observed far above expected = one limit shared by workers; "
            "close to expected = per-worker limit"
        )

    print("\n5. 429 rate by gap since that worker's previous request")
    lo = 0
    for b in GAP_BUCKETS[1:]:
        n = gap_requests[b]
        label = f"{lo}-{b:.0f}s" if b != math.inf else f">={lo}s"
        print(f"  {label:>9}  {n:7} req  {gap_limited[b]:6} 429  {_rate(gap_limited[b], n)}")
        lo = b

    print(f"\n6. pages given up after exhausting 429 retries: {sum(gave_up.values())}")

    if retry_after or first_bodies:
        print("\nwhat the 429 responses carried (new log format only)")
        for value, n in retry_after.most_common(10):
            print(f"  Retry-After {value}: {n}")
        for value, n in limit_headers.most_common(5):
            print(f"  limit headers {value}: {n}")
        for body in first_bodies:
            print(f"  body {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=Path, default=Path("./logs"))
    parser.add_argument(
        "--window", type=float, default=10.0, help="seconds for cross-worker clustering"
    )
    args = parser.parse_args()
    report(args.log_dir, args.window)


if __name__ == "__main__":
    main()
