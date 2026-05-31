import os
import signal
import threading

_flag = threading.Event()
STOP_FILE_ENV = "AUTOPARAM_STOP_FILE"


def install_shutdown_handler(signum=signal.SIGUSR1):
    def _handler(_signum, _frame):
        _flag.set()
    signal.signal(signum, _handler)


def shutdown_requested() -> bool:
    if _flag.is_set():
        return True
    path = os.environ.get(STOP_FILE_ENV)
    if path and os.path.exists(path):
        _flag.set()
        return True
    return False


def reset():
    _flag.clear()
