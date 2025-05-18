"""
Helper which record where time is spent.
"""

import time
import atexit
from typing import Optional
import json
from collections import defaultdict


class TimerRegistry:
    _instance: Optional["TimerRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimerRegistry, cls).__new__(cls)
            cls._instance.entries = defaultdict(float)
            atexit.register(cls._instance.print_output)
        return cls._instance

    def add_entry(self, name, value):
        self.entries[name] += value

    def get_entry(self, name):
        return self.entries.get(name)

    def get_all_entries(self):
        return self.entries.copy()

    def print_output(self):
        print(json.dumps(self.entries, indent=4))


class Timer:
    def __init__(self, name):
        self.name = name
        self.registry = TimerRegistry()
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.registry.add_entry(self.name, time.time() - self.start)
        return False

    def add_value(self, value):
        self.value_to_store = value
        self.registry.add_entry(self.name, value)
