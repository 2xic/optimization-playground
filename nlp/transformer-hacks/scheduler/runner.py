"""
Round-robin time-slicing scheduler.

Usage:
    python -m scheduler.runner [--config scheduler/scheduler.yaml]
"""
import argparse
import atexit
import hashlib
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
import contextlib

PKG_DIR = Path(__file__).resolve().parent
APP_DIR = PKG_DIR.parent
LOG_DIR = PKG_DIR / "logs"
STATUS_FILE = PKG_DIR / "status.json"
DEFAULT_CONFIG = PKG_DIR / "scheduler.yaml"
SCHEDULER_LOG = PKG_DIR / "scheduler.log"
CURRENT_LOG_LINK = "/tmp/autoparam_current.log"
INBOX_DIR = PKG_DIR / "inbox"
INBOX_FAILED = INBOX_DIR / "failed"
INBOX_LEDGER = PKG_DIR / "inbox_processed.txt"

_logger = None


def log(msg):
    line = f"[scheduler {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _logger is not None:
        with contextlib.suppress(Exception):
            _logger.info(msg)


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
            with contextlib.suppress(Exception):
                old.unlink()
    except Exception:
        pass


def detect_gpus_total() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return len([ln for ln in out.splitlines() if ln.strip()])
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
    cfg.setdefault("version", 0)
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
    np_env = (j.get("env") or {}).get("NUM_PROCESS")
    if np_env is not None:
        try:
            np_val = int(np_env)
        except (TypeError, ValueError):
            raise SystemExit(f"job '{j['name']}' has non-integer NUM_PROCESS: {np_env!r}")
        if np_val <= 0:
            raise SystemExit(f"job '{j['name']}' has non-positive NUM_PROCESS")
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
        hits = [ln for ln in lines if any(k in ln for k in keywords)]
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
_grace = {"seconds": 120}


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


def _reload_handler(signum, _frame):
    log(f"received signal {signum} — cooperative stop of active job for code reload")
    proc = _active_proc.get("proc")
    if proc is not None and proc.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGUSR1)
        try:
            proc.wait(timeout=_grace["seconds"])
        except Exception:
            log("grace expired — SIGKILL before reload")
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
    launcher = job.get("launcher") or ["python3", "-u"]
    cmd = [str(x) for x in launcher] + [job["script"]] + [str(a) for a in job["args"]] + extra
    return cmd, env, cuda_visible


def _point_current_log(log_path: Path):
    with contextlib.suppress(OSError):
        if os.path.islink(CURRENT_LOG_LINK) or os.path.exists(CURRENT_LOG_LINK):
            os.unlink(CURRENT_LOG_LINK)
        os.symlink(str(log_path), CURRENT_LOG_LINK)


def run_slot(job: dict, status: Status, mem_threshold: int, cleanup_timeout: int,
             max_quick_exits: int, quick_exit_seconds: int, keep_job_logs: int,
             stop_grace_seconds: int, preempt_check=None, once=False):
    slot_deadline = time.time() + job["slot_minutes"] * 60
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _prune_job_logs(job["name"], keep_job_logs)
    quick_exits = 0
    preempted = False

    while time.time() < slot_deadline:
        if not wait_gpu_free(job, mem_threshold, cleanup_timeout):
            log(f"skip launch {job['name']}: GPUs busy after {cleanup_timeout}s: {gpu_memory_used_mb()}")
            break
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
        _active_proc["pgid"] = proc.pid
        _point_current_log(log_path)
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
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGUSR1)
                    sigusr1_sent = True
                    sigusr1_time = now
                elif not sigusr1_sent and preempt_check is not None and preempt_check():
                    log(f"inbox job queued — preempting {job['name']} early (SIGUSR1 pgid {proc.pid})")
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGUSR1)
                    sigusr1_sent = True
                    sigusr1_time = now
                    preempted = True
                elif sigusr1_sent and not killed and now - sigusr1_time >= stop_grace_seconds:
                    log(
                        f"grace period {stop_grace_seconds}s expired — escalating to SIGKILL on pgid {proc.pid}"
                    )
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(proc.pid, signal.SIGKILL)
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

        if once and rc == 0:
            log(f"{job['name']} completed (once mode) — not relaunching")
            cleanup(job, mem_threshold, cleanup_timeout)
            return False, preempted

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


def wait_gpu_free(job: dict, mem_threshold: int, cleanup_timeout: int):
    deadline = time.time() + cleanup_timeout
    while time.time() < deadline:
        mem = gpu_memory_used_mb()
        n = job["gpus"] if job["gpus"] and job["gpus"] > 0 else len(mem)
        if mem and all(m < mem_threshold for m in mem[:n]):
            return True
        time.sleep(2)
    return False


def gpu_compute_pids():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,nounits,noheader"],
            text=True,
        )
        return sorted({int(x.strip()) for x in out.splitlines() if x.strip()})
    except Exception as e:
        log(f"nvidia-smi compute-apps query failed: {e}")
        return []


def our_gpu_pids(pgid):
    if not pgid:
        return []
    ours = []
    for pid in gpu_compute_pids():
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.getpgid(pid) == pgid:
                ours.append(pid)
    return ours


def kill_our_gpu_pids(pgid, sig):
    for pid in our_gpu_pids(pgid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.kill(pid, sig)


def cleanup(job: dict, mem_threshold: int, cleanup_timeout: int):
    script_basename = os.path.basename(job["script"])
    pattern = script_basename.replace(".py", "")
    pgid = _active_proc.get("pgid")
    log(f"cleanup after {job['name']}: pkill -f {pattern} + reap job pgid {pgid} + wait for GPU memory < {mem_threshold} MB")
    subprocess.run(["pkill", "-9", "-f", pattern], check=False)
    if pgid:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, signal.SIGTERM)
    kill_our_gpu_pids(pgid, signal.SIGTERM)
    deadline = time.time() + cleanup_timeout
    escalated = False
    while time.time() < deadline:
        mem = gpu_memory_used_mb()
        n = job["gpus"] if job["gpus"] and job["gpus"] > 0 else len(mem)
        if mem and all(m < mem_threshold for m in mem[:n]):
            log(f"GPUs clean: {mem[:n]} MB")
            return
        if not our_gpu_pids(pgid):
            log("job GPU pids gone; remaining memory held by other processes — leaving them alone")
            return
        if not escalated and time.time() - (deadline - cleanup_timeout) >= cleanup_timeout / 2:
            log(f"cleanup: job GPU pids still alive {our_gpu_pids(pgid)} — escalating to SIGKILL")
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pgid, signal.SIGKILL)
            kill_our_gpu_pids(pgid, signal.SIGKILL)
            escalated = True
        time.sleep(2)
    log(f"WARNING: cleanup timeout — job GPU pids {our_gpu_pids(pgid)} mem {gpu_memory_used_mb()}")


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_load() -> set:
    if not INBOX_LEDGER.exists():
        return set()
    return set(INBOX_LEDGER.read_text().split())


def _ledger_add(h: str):
    with open(INBOX_LEDGER, "a") as f:
        f.write(h + "\n")


def next_inbox_file():
    if not INBOX_DIR.exists():
        return None
    ledger = _ledger_load()
    mtimes = []
    for p in INBOX_DIR.glob("*.yaml"):
        try:
            h = _file_hash(p)
        except FileNotFoundError:
            continue
        if h in ledger:
            log(f"[inbox] {p.name} already processed (rsync re-drop) — purging")
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
            continue
        try:
            mtimes.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue
    if not mtimes:
        return None
    return min(mtimes, key=lambda t: t[0])[1]


def _move_inbox(path: Path, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    target = dest_dir / f"{ts}-{path.name}"
    try:
        path.replace(target)
        return True
    except Exception as e:
        log(f"[inbox] move {path.name} failed: {e}")
        return False


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
    _point_current_log(log_path)
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
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGUSR1)
                sigusr1_sent = True
                sigusr1_time = now
            elif sigusr1_sent and not killed and now - sigusr1_time >= stop_grace_seconds:
                log(f"[inbox] grace expired — SIGKILL pgid {proc.pid}")
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
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
    h = _file_hash(path)
    all_ok = True
    for j in jobs:
        ok = run_inbox_job(
            j, status, cfg["mem_free_threshold_mb"], cfg["cleanup_timeout_sec"],
            cfg["keep_job_logs"], cfg["stop_grace_seconds"],
        )
        all_ok = all_ok and ok
    if all_ok:
        _ledger_add(h)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        log(f"=== inbox done: {path.name} → ledger (ok) ===")
    else:
        moved = _move_inbox(path, INBOX_FAILED)
        log(f"=== inbox done: {path.name} → failed/ (had failures{'' if moved else ', MOVE FAILED'}) ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="exit cleanly after this many job slots (0 = run forever)")
    parser.add_argument("--once", action="store_true",
                        help="run each enabled job once until it completes, then exit")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    _init_logger()

    cfg = load_config(args.config)
    start_version = cfg["version"]
    gpus_total = detect_gpus_total()
    validate(cfg, gpus_total)
    log(f"detected {gpus_total} GPUs; {len(cfg['jobs'])} jobs configured")

    _grace["seconds"] = cfg["stop_grace_seconds"]
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGHUP, _reload_handler)
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
    done = set()
    inbox_enabled = cfg.get("inbox", True)
    cycles = 0
    while True:
        inbox = next_inbox_file() if inbox_enabled else None
        if inbox is not None:
            process_inbox_file(inbox, status, cfg, gpus_total)
            continue
        now = time.time()
        order = {j["name"]: k for k, j in enumerate(jobs)}
        eligible = [j for j in jobs if quarantine_until.get(j["name"], 0.0) <= now]
        if args.once:
            eligible = [j for j in eligible if j["name"] not in done]
            if not eligible:
                log("all jobs completed (once mode) — exiting")
                return
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
                preempt_check=lambda: (inbox_enabled and next_inbox_file() is not None)
                or _config_mtime(args.config) not in (None, cfg_mtime),
                once=args.once)
        except Exception as e:
            log(f"run_slot raised: {e}")
            cleanup(job, cfg["mem_free_threshold_mb"], cfg["cleanup_timeout_sec"])
            abandoned, preempted = True, False
        if preempted:
            if _config_mtime(args.config) not in (None, cfg_mtime):
                log(f"{job['name']} preempted by config change — reloading")
            else:
                log(f"{job['name']} preempted by inbox — will resume after draining inbox")
                continue
        runs[job["name"]] += 1
        if abandoned:
            quarantine_until[job["name"]] = time.time() + quarantine_sec
            log(f"quarantining {job['name']} for {cfg['quarantine_hours']}h")
        elif args.once:
            done.add(job["name"])
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
                raise SystemExit(f"invalid config on reload: {e}") from e
            else:
                if new_cfg["version"] != start_version:
                    log(f"version {start_version} → {new_cfg['version']}: cooperative reload for new code")
                    _reload_handler(0, None)
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
