import time
from collections import defaultdict

class TimeIt:
    def __init__(self, name) -> None:
        self.name = name
        self.times = []

    def __enter__(self):
        return self

    def __call__( self):
        self.start = time.time_ns()
        return self

    def __exit__(self, *_args):
        self.times.append(time.time_ns() - self.start)

    def __repr__(self) -> str:
        return self.__str__()
