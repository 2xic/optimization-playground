"""
Round-robin time-slicing scheduler.

Usage:
    python -m scheduler.runner [--config scheduler/scheduler.yaml]
"""
import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import yaml

PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent
LOG_DIR = PKG_DIR / "logs"
STATUS_FILE = PKG_DIR / "status.json"
DEFAULT_CONFIG = PKG_DIR / "scheduler.yaml"
SCHEDULER_LOG = PKG_DIR / "scheduler.log"
INBOX_DIR = PKG_DIR / "inbox"
INBOX_DONE = INBOX_DIR / "done"
INBOX_FAILED = INBOX_DIR / "failed"

_logger = None


def log(msg):
    line = f"[scheduler {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _logger is not None:
        try:
            _logger.info(msg)
        except Exception:
            pass


def _init_logger():
    global _logger
    logger = logging.getLogger("scheduler")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = RotatingFileHandler(SCHEDULER_LOG, maxBytes=10 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    _logger = logger


def _prune_job_logs(job_name: str, keep: int):
    try:
        files = sorted(LOG_DIR.glob(f"{job_name}-*.log"))
        for old in files[:-keep]:
            try:
                old.unlink()
            except Exception:
                pass
    except Exception:
        pass


def detect_gpus_total() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([l for l in out.splitlines() if l.strip()])
    except Exception as e:
        log(f"nvidia-smi -L failed: {e} — assuming 0 GPUs")
        return 0


def gpu_memory_used_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
            text=True,
        )
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception as e:
        log(f"nvidia-smi memory query failed: {e}")
        return []


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not cfg.get("jobs"):
        raise SystemExit("config must define non-empty 'jobs'")
    cfg.setdefault("mem_free_threshold_mb", 500)
    cfg.setdefault("cleanup_timeout_sec", 60)
    cfg.setdefault("max_quick_exits", 3)
    cfg.setdefault("quick_exit_seconds", 60)
    cfg.setdefault("quarantine_hours", 1.0)
    cfg.setdefault("keep_job_logs", 20)
    cfg.setdefault("stop_grace_seconds", 1800)
    return cfg


def _validate_job(j: dict, gpus_total: int):
    for required in ("name", "script", "gpus"):
        if required not in j:
            raise SystemExit(f"job missing field '{required}': {j}")
    if j["gpus"] == -1:
        j["gpus"] = gpus_total
    if j["gpus"] <= 0 or j["gpus"] > gpus_total:
        raise SystemExit(
            f"job '{j['name']}' requests {j['gpus']} GPUs; only {gpus_total} available"
        )
    if "slot_minutes" not in j:
        raise SystemExit(f"job '{j['name']}' missing slot_minutes")
    j["slot_minutes"] = float(j["slot_minutes"])
    if j["slot_minutes"] <= 0:
        raise SystemExit(f"job '{j['name']}' has non-positive slot_minutes")
    j.setdefault("args", [])
    j.setdefault("pass_nproc_per_node", True)
    j.setdefault("enabled", True)
    j["weight"] = int(j.get("weight", 1))
    if j["weight"] < 1:
        raise SystemExit(f"job '{j['name']}' has weight < 1")


def validate(cfg: dict, gpus_total: int):
    if gpus_total <= 0:
        raise SystemExit("no GPUs detected via nvidia-smi -L")
    for j in cfg["jobs"]:
        _validate_job(j, gpus_total)


def _config_mtime(path: Path):
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _build_jobs(cfg: dict):
    return [j for j in cfg["jobs"] if j["enabled"]]


class Status:
    def __init__(self):
        self.history = deque(maxlen=50)
        self.current = None

    def write(self):
        payload = {
            "current": self.current,
            "history": list(self.history),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = STATUS_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(STATUS_FILE)


def tail_log(path: Path, n: int = 50) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 16384)
            f.seek(size - chunk)
            data = f.read().decode(errors="replace")
        return "\n".join(data.splitlines()[-n:])
    except Exception as e:
        return f"<log tail failed: {e}>"


def extract_error_excerpt(path: Path, tail_lines: int = 30) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 65536)
            f.seek(size - chunk)
            data = f.read().decode(errors="replace")
        lines = data.splitlines()
        for i in range(len(lines) - 1, -1, -1):
            if "Traceback (most recent call last)" in lines[i]:
                return "\n".join(lines[i:])
        keywords = ("Error", "error:", "Exception", "FAILED", "CUDA out of memory", "RuntimeError")
        hits = [l for l in lines if any(k in l for k in keywords)]
        if hits:
            return "\n".join(hits[-tail_lines:])
        return "\n".join(lines[-tail_lines:])
    except Exception as e:
        return f"<error extract failed: {e}>"


def log_error_excerpt(job_name: str, log_path: Path, rc: int, ran_seconds: float):
    excerpt = extract_error_excerpt(log_path)
    if not excerpt.strip():
        return
    log(f"!!! {job_name} failure detail (rc={rc}, ran={ran_seconds:.0f}s, log={log_path}):")
    for line in excerpt.splitlines():
        log(f"    | {line}")


_active_proc = {"proc": None}


def _terminate_active(sig=signal.SIGTERM):
    proc = _active_proc.get("proc")
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    except Exception as e:
        log(f"killpg({sig}) failed: {e}")


def _shutdown_handler(signum, _frame):
    log(f"received signal {signum} — terminating child and exiting")
    _terminate_active(signal.SIGTERM)
    try:
        proc = _active_proc.get("proc")
        if proc is not None:
            proc.wait(timeout=30)
    except Exception:
        _terminate_active(signal.SIGKILL)
    sys.exit(0)


def build_cmd_env(job: dict, gpus: int, slot_minutes: float):
    cuda_visible = ",".join(str(i) for i in range(gpus))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    env.setdefault("TRAINING_TIME_MINUTES", str(max(1, int(slot_minutes) - 5)))
    for k, v in (job.get("env") or {}).items():
        env[str(k)] = str(v)
    extra = []
    if job.get("pass_nproc_per_node", True) and not any(
        str(a).startswith("--nproc_per_node") or str(a).startswith("--nproc-per-node")
        for a in job["args"]
    ):
        extra = [f"--nproc-per-node={gpus}"]
    cmd = ["python3", "-u", job["script"]] + [str(a) for a in job["args"]] + extra
    return cmd, env, cuda_visible


def run_slot(job: dict, status: Status, mem_threshold: int, cleanup_timeout: int,
             max_quick_exits: int, quick_exit_seconds: int, keep_job_logs: int,
             stop_grace_seconds: int, preempt_check=None):
    slot_deadline = time.time() + job["slot_minutes"] * 60
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_job_logs(job["name"], keep_job_logs)
    quick_exits = 0
    preempted = False

    while time.time() < slot_deadline:
        ts = time.strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"{job['name']}-{ts}.log"
        cmd, env, cuda_visible = build_cmd_env(job, job["gpus"], job["slot_minutes"])
        log(f"launching {job['name']}: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={cuda_visible})")
        log(f"  log: {log_path}")

        started = datetime.now().isoformat(timespec="seconds")
        launch_time = time.time()
        with open(log_path, "ab") as lf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(APP_DIR),
                env=env,
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        _active_proc["proc"] = proc
        status.current = {
            "job": job["name"],
            "pid": proc.pid,
            "started_at": started,
            "slot_deadline": datetime.fromtimestamp(slot_deadline).isoformat(timespec="seconds"),
            "log": str(log_path),
        }
        status.write()

        sigusr1_sent = False
        sigusr1_time = None
        killed = False
        while True:
            try:
                rc = proc.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                now = time.time()
                if not sigusr1_sent and now >= slot_deadline:
                    log(f"slot deadline reached — sending SIGUSR1 to pgid {proc.pid} (cooperative)")
                    try:
                        os.killpg(proc.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                    sigusr1_sent = True
                    sigusr1_time = now
                elif not sigusr1_sent and preempt_check is not None and preempt_check():
                    log(f"inbox job queued — preempting {job['name']} early (SIGUSR1 pgid {proc.pid})")
                    try:
                        os.killpg(proc.pid, signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                    sigusr1_sent = True
                    sigusr1_time = now
                    preempted = True
                elif sigusr1_sent and not killed and now - sigusr1_time >= stop_grace_seconds:
                    log(
                        f"grace period {stop_grace_seconds}s expired — escalating to SIGKILL on pgid {proc.pid}"
                    )
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    killed = True

        _active_proc["proc"] = None
        ended = datetime.now().isoformat(timespec="seconds")
        entry = {
            "job": job["name"],
            "started": started,
            "ended": ended,
            "exit_code": rc,
            "error": tail_log(log_path) if rc != 0 else None,
            "cooperative_shutdown": sigusr1_sent,
        }
        status.history.append(entry)
        status.current = None
        status.write()
        ran_seconds = time.time() - launch_time
        log(f"{job['name']} exited rc={rc} after {ran_seconds:.0f}s")

        if rc != 0 or ran_seconds < quick_exit_seconds:
            log_error_excerpt(job["name"], log_path, rc, ran_seconds)

        if sigusr1_sent:
            break

        if ran_seconds < quick_exit_seconds:
            quick_exits += 1
            log(f"quick exit ({quick_exits}/{max_quick_exits}) — ran <{quick_exit_seconds}s")
            if quick_exits >= max_quick_exits:
                log(f"hit {max_quick_exits} quick exits in a row — abandoning slot, advancing to next job")
                cleanup(job, mem_threshold, cleanup_timeout)
                return True, False
        else:
            quick_exits = 0

    cleanup(job, mem_threshold, cleanup_timeout)
    return False, preempted


def cleanup(job: dict, mem_threshold: int, cleanup_timeout: int):
    script_basename = os.path.basename(job["script"])
    pattern = script_basename.replace(".py", "")
    log(f"cleanup after {job['name']}: pkill -f {pattern} + wait for GPU memory < {mem_threshold} MB")
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    deadline = time.time() + cleanup_timeout
    while time.time() < deadline:
        mem = gpu_memory_used_mb()
        n = job["gpus"] if job["gpus"] and job["gpus"] > 0 else len(mem)
        if mem and all(m < mem_threshold for m in mem[:n]):
            log(f"GPUs clean: {mem[:n]} MB")
            return
        time.sleep(2)
    log(f"WARNING: cleanup timeout — GPU memory still: {gpu_memory_used_mb()}")


def next_inbox_file():
    if not INBOX_DIR.exists():
        return None
    mtimes = []
    for p in INBOX_DIR.glob("*.yaml"):
        try:
            mtimes.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    if not mtimes:
        return None
    return min(mtimes, key=lambda t: t[0])[1]


def _move_inbox(path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    target = dest_dir / f"{ts}-{path.name}"
    try:
        path.replace(target)
    except Exception as e:
        log(f"[inbox] move {path.name} failed: {e}")


def run_inbox_job(job: dict, status: Status, mem_threshold: int, cleanup_timeout: int,
                  keep_job_logs: int, stop_grace_seconds: int) -> bool:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_job_logs(job["name"], keep_job_logs)
    deadline = time.time() + job["slot_minutes"] * 60
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{job['name']}-{ts}.log"
    cmd, env, cuda_visible = build_cmd_env(job, job["gpus"], job["slot_minutes"])
    log(f"[inbox] launching {job['name']}: {' '.join(cmd)} (CUDA_VISIBLE_DEVICES={cuda_visible})")
    log(f"  log: {log_path}")

    started = datetime.now().isoformat(timespec="seconds")
    launch_time = time.time()
    with open(log_path, "ab") as lf:
        proc = subprocess.Popen(
            cmd, cwd=str(APP_DIR), env=env, stdout=lf,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    _active_proc["proc"] = proc
    status.current = {
        "job": f"inbox:{job['name']}",
        "pid": proc.pid,
        "started_at": started,
        "slot_deadline": datetime.fromtimestamp(deadline).isoformat(timespec="seconds"),
        "log": str(log_path),
    }
    status.write()

    sigusr1_sent = False
    sigusr1_time = None
    killed = False
    while True:
        try:
            rc = proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if not sigusr1_sent and now >= deadline:
                log(f"[inbox] {job['name']} hit slot cap — SIGUSR1 pgid {proc.pid}")
                try:
                    os.killpg(proc.pid, signal.SIGUSR1)
                except ProcessLookupError:
                    pass
                sigusr1_sent = True
                sigusr1_time = now
            elif sigusr1_sent and not killed and now - sigusr1_time >= stop_grace_seconds:
                log(f"[inbox] grace expired — SIGKILL pgid {proc.pid}")
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                killed = True

    _active_proc["proc"] = None
    ran_seconds = time.time() - launch_time
    ended = datetime.now().isoformat(timespec="seconds")
    status.history.append({
        "job": f"inbox:{job['name']}",
        "started": started,
        "ended": ended,
        "exit_code": rc,
        "error": tail_log(log_path) if rc != 0 else None,
        "cooperative_shutdown": sigusr1_sent,
    })
    status.current = None
    status.write()
    log(f"[inbox] {job['name']} exited rc={rc} after {ran_seconds:.0f}s")
    if rc != 0:
        log_error_excerpt(job["name"], log_path, rc, ran_seconds)
    cleanup(job, mem_threshold, cleanup_timeout)
    return rc == 0


def process_inbox_file(path: Path, status: Status, cfg: dict, gpus_total: int):
    log(f"=== inbox steal: {path.name} ===")
    try:
        data = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        log(f"[inbox] {path.name} removed before processing — skipping")
        return
    except Exception as e:
        log(f"[inbox] failed to parse {path.name}: {e} — moving to failed/")
        _move_inbox(path, INBOX_FAILED)
        return
    if isinstance(data, dict) and "jobs" in data:
        jobs = data["jobs"]
    elif isinstance(data, list):
        jobs = data
    elif isinstance(data, dict):
        jobs = [data]
    else:
        log(f"[inbox] {path.name} has no jobs — moving to failed/")
        _move_inbox(path, INBOX_FAILED)
        return
    try:
        for j in jobs:
            j.setdefault("gpus", -1)
            j.setdefault("slot_minutes", 30)
            _validate_job(j, gpus_total)
    except SystemExit as e:
        log(f"[inbox] invalid job in {path.name}: {e} — moving to failed/")
        _move_inbox(path, INBOX_FAILED)
        return
    all_ok = True
    for j in jobs:
        ok = run_inbox_job(
            j, status, cfg["mem_free_threshold_mb"], cfg["cleanup_timeout_sec"],
            cfg["keep_job_logs"], cfg["stop_grace_seconds"],
        )
        all_ok = all_ok and ok
    dest = INBOX_DONE if all_ok else INBOX_FAILED
    _move_inbox(path, dest)
    log(f"=== inbox done: {path.name} → {dest.name}/ ({'ok' if all_ok else 'had failures'}) ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="exit cleanly after this many job slots (0 = run forever)")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    _init_logger()

    cfg = load_config(args.config)
    gpus_total = detect_gpus_total()
    validate(cfg, gpus_total)
    log(f"detected {gpus_total} GPUs; {len(cfg['jobs'])} jobs configured")

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)
    atexit.register(_terminate_active, signal.SIGTERM)

    status = Status()
    status.write()
    jobs = _build_jobs(cfg)
    if not jobs:
        raise SystemExit("no enabled jobs")
    cfg_mtime = _config_mtime(args.config)
    quarantine_until = {j["name"]: 0.0 for j in jobs}
    quarantine_sec = cfg["quarantine_hours"] * 3600
    runs = {j["name"]: 0 for j in jobs}
    cycles = 0
    while True:
        inbox = next_inbox_file()
        if inbox is not None:
            process_inbox_file(inbox, status, cfg, gpus_total)
            continue
        now = time.time()
        order = {j["name"]: k for k, j in enumerate(jobs)}
        eligible = [j for j in jobs if quarantine_until.get(j["name"], 0.0) <= now]
        if not eligible:
            sleep_for = min(quarantine_until[j["name"]] for j in jobs) - now
            log(f"all jobs quarantined — sleeping {sleep_for:.0f}s")
            time.sleep(max(sleep_for, 1))
            continue
        job = min(eligible, key=lambda j: (runs[j["name"]] / j["weight"], order[j["name"]]))
        log(f"=== slot start: {job['name']} ({job['slot_minutes']:.1f}m) ===")
        try:
            abandoned, preempted = run_slot(
                job, status, cfg["mem_free_threshold_mb"], cfg["cleanup_timeout_sec"],
                cfg["max_quick_exits"], cfg["quick_exit_seconds"],
                cfg["keep_job_logs"], cfg["stop_grace_seconds"],
                preempt_check=lambda: next_inbox_file() is not None)
        except Exception as e:
            log(f"run_slot raised: {e}")
            cleanup(job, cfg["mem_free_threshold_mb"], cfg["cleanup_timeout_sec"])
            abandoned, preempted = True, False
        if preempted:
            log(f"{job['name']} preempted by inbox — will resume after draining inbox")
            continue
        runs[job["name"]] += 1
        if abandoned:
            quarantine_until[job["name"]] = time.time() + quarantine_sec
            log(f"quarantining {job['name']} for {cfg['quarantine_hours']}h")
        cycles += 1
        if args.max_cycles and cycles >= args.max_cycles:
            log(f"reached --max-cycles={args.max_cycles}, exiting cleanly")
            return

        new_mtime = _config_mtime(args.config)
        if new_mtime is not None and new_mtime != cfg_mtime:
            try:
                new_cfg = load_config(args.config)
                validate(new_cfg, gpus_total)
                new_jobs = _build_jobs(new_cfg)
                if not new_jobs:
                    raise SystemExit("no enabled jobs")
            except Exception as e:
                log(f"FATAL: config reload failed — fix scheduler.yaml and restart: {e}")
                raise SystemExit(f"invalid config on reload: {e}")
            else:
                base = min(runs.values()) if runs else 0
                cfg, jobs, cfg_mtime = new_cfg, new_jobs, new_mtime
                quarantine_sec = cfg["quarantine_hours"] * 3600
                quarantine_until = {
                    j["name"]: quarantine_until.get(j["name"], 0.0) for j in jobs
                }
                runs = {j["name"]: runs.get(j["name"], base) for j in jobs}
                added = [j["name"] for j in jobs if j["name"] not in order]
                log(f"config reloaded — {len(jobs)} jobs"
                    + (f", new: {', '.join(added)}" if added else ""))


if __name__ == "__main__":
    main()
