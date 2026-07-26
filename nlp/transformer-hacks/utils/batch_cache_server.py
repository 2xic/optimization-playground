import asyncio
import logging
import os
from collections import OrderedDict

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("batch_cache")

import aiohttp
from aiohttp import web
from aiohttp_retry import RetryClient, JitterRetry
import numpy as np
import ormsgpack

ORIGIN = os.environ["WEB_DATALOADER"].rstrip("/")
PORT = int(os.environ.get("BATCH_CACHE_PORT", 8899))
MAX_BYTES = int(os.environ.get("BATCH_CACHE_MAX_BYTES", 8 * 1024**3))

_cache = OrderedDict()
_cache_bytes = 0
_inflight = {}


def _npdt(code):
    return np.float32 if code == "f4" else np.int64


def _build_blob(rows, cols, dtypes, dataset):
    header_cols, dt_strs, shapes, nbytes, buffers = [], [], [], [], []
    for col, code in zip(cols, dtypes):
        arr = np.ascontiguousarray(
            np.array([item[col] for item in rows], dtype=_npdt(code))
        )
        b = arr.tobytes()
        header_cols.append(col)
        dt_strs.append(arr.dtype.str)
        shapes.append(list(arr.shape))
        nbytes.append(len(b))
        buffers.append(b)
    header = ormsgpack.packb(
        {
            "dataset": dataset,
            "cols": header_cols,
            "dtypes": dt_strs,
            "shapes": shapes,
            "nbytes": nbytes,
        }
    )
    return len(header).to_bytes(4, "big") + header + b"".join(buffers)


def _empty_blob(cols, dtypes, dataset):
    return _build_blob([], cols, dtypes, dataset)


def _cache_put(key, blob):
    global _cache_bytes
    _cache[key] = blob
    _cache_bytes += len(blob)
    while _cache_bytes > MAX_BYTES and len(_cache) > 1:
        _, old = _cache.popitem(last=False)
        _cache_bytes -= len(old)


async def _fetch_origin(session, ds, split, start, end, cols):
    url = (
        f"{ORIGIN}/datasets/{ds}/{split}/get"
        f"?start={start}&end={end}&columns={','.join(cols)}"
    )
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


async def handle_getb(request):
    ds = request.match_info["ds"]
    split = request.match_info["split"]
    start = request.query["start"]
    end = request.query["end"]
    cols = request.query["columns"].split(",")
    dtypes = request.query.get("dtypes", ",".join("i8" for _ in cols)).split(",")
    key = f"{ds}/{split}/{start}/{end}/{','.join(cols)}"

    blob = _cache.get(key)
    if blob is not None:
        _cache.move_to_end(key)
        return web.Response(body=blob, content_type="application/octet-stream")

    fut = _inflight.get(key)
    if fut is not None:
        blob = await fut
        return web.Response(body=blob, content_type="application/octet-stream")

    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    _inflight[key] = fut
    try:
        try:
            content = await _fetch_origin(
                request.app["session"], ds, split, start, end, cols
            )
            rows = ormsgpack.unpackb(content)
            blob = _build_blob(rows, cols, dtypes, ds)
            _cache_put(key, blob)
        except Exception:
            logger.exception("origin/decode failed key=%s", key)
            blob = _empty_blob(cols, dtypes, ds)
        fut.set_result(blob)
    finally:
        _inflight.pop(key, None)
    return web.Response(body=blob, content_type="application/octet-stream")


async def _on_startup(app):
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=30)
    retry = JitterRetry(
        attempts=8,
        start_timeout=1.0,
        max_timeout=30.0,
        factor=2.0,
        random_interval_size=2.0,
        retry_all_server_errors=True,
    )
    app["session"] = RetryClient(
        connector=connector, timeout=timeout, retry_options=retry
    )


async def _on_cleanup(app):
    await app["session"].close()


def main():
    app = web.Application()
    app.router.add_get("/datasets/{ds}/{split}/getb", handle_getb)
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)


if __name__ == "__main__":
    main()
