"""Continuous 10 Hz CPU-arrival measurement, including queueing."""

import math

from queue import Empty, Full, Queue

import statistics

import threading

import time


def percentile(values, fraction=0.95):
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def summarize_sequence(rows, expected_frames, fps, errors=()):
    completed = len(rows)
    if not completed:
        return {
            "frames": 0,
            "expected_frames": expected_frames,
            "complete": False,
            "realtime_pass": False,
            "errors": list(errors),
        }
    latencies = [row["latency_seconds"] for row in rows]
    service = [row["service_seconds"] for row in rows]
    waits = [row["queue_seconds"] for row in rows]
    window = max(1, completed // 4)
    growth = statistics.median(waits[-window:]) - statistics.median(waits[:window])
    duration = max(expected_frames / fps, rows[-1]["completion_seconds"])
    throughput = completed / duration
    source_late = sum(row["source_late_seconds"] > 0 for row in rows)
    backpressure = any(
        row["input_backpressure"] or row.get("output_backpressure", False)
        for row in rows
    )
    complete = completed == expected_frames and not errors
    p95 = percentile(latencies)
    period = 1.0 / fps
    return {
        "frames": completed,
        "expected_frames": expected_frames,
        "complete": complete,
        "offered_fps": fps,
        "sustained_fps": throughput,
        "service_fps": completed / sum(service) if sum(service) else None,
        "p95_seconds": p95,
        "max_seconds": max(latencies),
        "queue_p95_seconds": percentile(waits),
        "queue_growth_seconds": growth,
        "growing_backlog": growth > period,
        "backlog_test": "last-quarter minus first-quarter median wait > one scan period",
        "max_pending_frames": max(row["pending_frames"] for row in rows),
        "source_late_frames": source_late,
        "queue_backpressure": backpressure,
        "disk_seconds": sum(row.get("disk_seconds", 0.0) for row in rows),
        "input_file_read_seconds": sum(row["input_file_read_seconds"] for row in rows),
        "realtime_pass": (
            complete
            and not source_late
            and not backpressure
            and growth <= period
            and p95 <= period
            and throughput + 1e-9 >= fps
        ),
        "quantile_method": "nearest-rank; all frames including first/reset; no trimming",
        "errors": list(errors),
    }


def measure_sequence(
    items,
    load,
    process,
    describe,
    sink,
    *,
    fps=10.0,
    prefetch_frames=2,
    max_pending_frames=32,
    lead_seconds=0.5,
):
    """process returns a complete result in CPU memory; every required sync/copy is inside it.

    Files are prefetched before scheduled arrival. A late source or bounded queue
    backpressure invalidates the real-time claim, while retaining all completed rates.
    An optional sink runs separately and is reported. With sink=None, results are
    consumed in CPU memory without a disk worker or output queue backpressure.
    """
    if fps <= 0 or prefetch_frames < 1 or max_pending_frames < 1 or lead_seconds < 0:
        raise ValueError("Invalid arrival or queue configuration")
    prefetched, arrived, outputs = (
        Queue(prefetch_frames),
        Queue(max_pending_frames),
        Queue(max_pending_frames),
    )
    done = object()
    stop = threading.Event()
    errors, rows = [], []
    releases = {}
    origin = time.perf_counter() + lead_seconds

    def put(queue, value):
        while not stop.is_set():
            try:
                queue.put(value, timeout=0.05)
                return True
            except Full:
                pass
        return False

    def get(queue):
        while not stop.is_set():
            try:
                return queue.get(timeout=0.05)
            except Empty:
                pass
        return done

    def preload():
        try:
            for index, item in enumerate(items):
                started = time.perf_counter()
                payload = load(item)
                ready = time.perf_counter()
                if not put(prefetched, (index, item, payload, ready, ready - started)):
                    return
        except BaseException as exc:
            errors.append(f"input: {type(exc).__name__}: {exc}")
        finally:
            put(prefetched, done)

    def release():
        while not stop.is_set():
            value = get(prefetched)
            if value is done:
                put(arrived, done)
                return
            index, item, payload, ready, read_seconds = value
            arrival = origin + index / fps
            if stop.wait(max(0.0, arrival - time.perf_counter())):
                return
            backpressure = arrived.full()
            attempted = time.perf_counter()
            if not put(
                arrived,
                (
                    index,
                    item,
                    payload,
                    arrival,
                    max(0.0, ready - arrival),
                    read_seconds,
                    backpressure,
                ),
            ):
                return
            releases[index] = (ready, attempted, time.perf_counter())

    def write_outputs():
        while not stop.is_set():
            value = get(outputs)
            if value is done:
                return
            result, item, row = value
            try:
                started = time.perf_counter()
                sink(result, item, row)
                row["disk_seconds"] = time.perf_counter() - started
            except BaseException as exc:
                errors.append(f"output: {type(exc).__name__}: {exc}")
                row["disk_error"] = str(exc)

    targets = [(preload, "prefetch"), (release, "arrival")]
    if sink is not None:
        targets.append((write_outputs, "disk"))
    workers = [
        threading.Thread(target=target, daemon=True, name=f"lidar-{name}")
        for target, name in targets
    ]
    for worker in workers:
        worker.start()
    try:
        while True:
            value = get(arrived)
            if value is done:
                break
            index, item, payload, arrival, source_late, read_seconds, backpressure = (
                value
            )
            pending = min(arrived.qsize(), len(items) - index - 1)
            started = time.perf_counter()
            result = process(payload, index)
            finished = time.perf_counter()
            row = {
                "frame": index,
                "arrival_seconds": index / fps,
                "service_start_seconds": started - origin,
                "completion_seconds": finished - origin,
                "service_seconds": finished - started,
                "queue_seconds": max(0.0, started - arrival),
                "latency_seconds": finished - arrival,
                "source_late_seconds": source_late,
                "input_file_read_seconds": read_seconds,
                "pending_frames": pending,
                "input_backpressure": backpressure,
                **describe(result, item),
            }
            rows.append(row)
            row["output_backpressure"] = sink is not None and outputs.full()
            if sink is not None and not put(outputs, (result, item, row)):
                break
        if sink is not None:
            put(outputs, done)
        for worker in workers:
            worker.join()
    except BaseException as exc:
        errors.append(f"processing: {type(exc).__name__}: {exc}")
        stop.set()
        for worker in workers:
            worker.join()  # Finalize disk/error accounting before releasing this sequence.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        stop.set()
    # Read release stamps only after the producer joins. Queue.put may wake the
    # consumer before its return; the two stamps bound the successful enqueue.
    for row in rows:
        stamp = releases.get(row["frame"])
        if stamp is not None:
            ready, attempted, enqueued = stamp
            row.update(
                input_ready_seconds=ready - origin,
                release_attempt_seconds=attempted - origin,
                enqueue_complete_seconds=enqueued - origin,
                enqueue_call_seconds=enqueued - attempted,
                release_late_seconds=max(
                    0.0, attempted - origin - row["arrival_seconds"]
                ),
                queue_after_enqueue_seconds=max(
                    0.0, row["service_start_seconds"] - (enqueued - origin)
                ),
            )
    return rows, summarize_sequence(rows, len(items), fps, errors)
