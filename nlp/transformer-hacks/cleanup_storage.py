import os
import sys
import json
import argparse
from datetime import date, datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

from utils.checkpoints import StorageBox, INDEX_PATH


def parse_args():
    p = argparse.ArgumentParser(description="Prune old checkpoints from storage box")
    p.add_argument("--keep-days", type=int, default=7)
    p.add_argument("--oldest-days", type=int, default=0, help="only delete the N oldest date dirs")
    p.add_argument("--before", type=str, default=None, help="delete date dirs strictly before YYYY-MM-DD")
    p.add_argument("--apply", action="store_true", help="actually delete (default dry-run)")
    p.add_argument("--keep-latest-per-run", action="store_true", help="never delete highest step of a run")
    return p.parse_args()


def load_protected(box):
    protected = set()
    tags_dir = "checkpoints/tags"
    try:
        tags = box.sftp.listdir(tags_dir)
    except FileNotFoundError:
        return protected
    for tag in tags:
        latest = f"{tags_dir}/{tag}/latest.json"
        try:
            with box.sftp.open(latest, "r") as f:
                data = json.loads(f.read().decode("utf-8"))
        except Exception as e:
            print(f"WARN: cannot read tag {tag}: {e}")
            continue
        path = data.get("path")
        if path:
            protected.add(path.rstrip("/"))
    return protected


def date_dirs(box):
    out = {}
    for name in box.sftp.listdir("checkpoints"):
        if name in ("tags",) or name.endswith(".ndjson"):
            continue
        try:
            d = datetime.strptime(name, "%Y-%m-%d").date()
        except ValueError:
            continue
        out[name] = d
    return out


def rmtree(box, path):
    for name in box.sftp.listdir(path):
        full = f"{path}/{name}"
        if box.is_directory(full):
            rmtree(box, full)
        else:
            box.sftp.remove(full)
    box.sftp.rmdir(path)


def latest_steps_per_run(box, entries):
    best = {}
    for e in entries:
        run = e.get("run_id")
        step = e.get("step", -1)
        if run is None:
            continue
        if step > best.get(run, (-1, None))[0]:
            best[run] = (step, e.get("path", "").rstrip("/"))
    return {v[1] for v in best.values() if v[1]}


def main():
    args = parse_args()
    box = StorageBox(
        host=os.environ["CHECKPOINT_STORAGE_BOX_HOST"],
        username=os.environ["CHECKPOINT_STORAGE_BOX_USERNAME"],
        password=os.environ["CHECKPOINT_STORAGE_BOX_PASSWORD"],
    )

    cutoff = date.today() - timedelta(days=args.keep_days)
    print("loading tags ...", flush=True)
    protected = load_protected(box)

    entries = []
    if args.keep_latest_per_run or args.apply:
        print("reading index.ndjson ...", flush=True)
        try:
            entries = box.iterate_index()
        except Exception:
            entries = []
    if args.keep_latest_per_run:
        protected |= latest_steps_per_run(box, entries)

    dirs = date_dirs(box)
    if args.before:
        before = datetime.strptime(args.before, "%Y-%m-%d").date()
        old = sorted(d for d, dt in dirs.items() if dt < before)
    elif args.oldest_days > 0:
        old = sorted(dirs, key=lambda d: dirs[d])[: args.oldest_days]
    else:
        old = sorted(d for d, dt in dirs.items() if dt < cutoff)

    print(f"today={date.today()} cutoff={cutoff} keep_days={args.keep_days} oldest_days={args.oldest_days} before={args.before}")
    print(f"date dirs: {len(dirs)}  targeted: {len(old)}  protected paths: {len(protected)}")
    print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    to_delete = []
    kept = []
    for name in old:
        base = f"checkpoints/{name}"
        print(f"scanning {base} ...", flush=True)
        for run in box.sftp.listdir(base):
            run_path = f"{base}/{run}"
            if not box.is_directory(run_path):
                continue
            for step in box.sftp.listdir(run_path):
                step_path = f"{run_path}/{step}"
                if step_path.rstrip("/") in protected:
                    kept.append(step_path)
                    continue
                to_delete.append(step_path)

    for p in to_delete:
        print(f"DELETE {p}")
    for p in kept:
        print(f"KEEP(protected) {p}")

    print(f"\n{len(to_delete)} step dirs to delete, {len(kept)} protected kept")

    if not args.apply:
        print("dry-run — nothing deleted. re-run with --apply")
        box.close()
        return

    deleted = set()
    for p in to_delete:
        try:
            rmtree(box, p)
            deleted.add(p)
            print(f"deleted {p}")
        except Exception as e:
            print(f"FAIL {p}: {e}")

    for name in old:
        base = f"checkpoints/{name}"
        for run in list(box.sftp.listdir(base)):
            run_path = f"{base}/{run}"
            try:
                if box.is_directory(run_path) and not box.sftp.listdir(run_path):
                    box.sftp.rmdir(run_path)
            except Exception:
                pass
        try:
            if not box.sftp.listdir(base):
                box.sftp.rmdir(base)
        except Exception:
            pass

    if deleted and entries:
        kept_entries = [e for e in entries if e.get("path", "").rstrip("/") not in deleted]
        buf = "".join(json.dumps(e) + "\n" for e in kept_entries).encode("utf-8")
        box.save_bytes(buf, INDEX_PATH)
        print(f"index rewritten: {len(entries)} -> {len(kept_entries)} entries")

    box.close()


if __name__ == "__main__":
    main()
