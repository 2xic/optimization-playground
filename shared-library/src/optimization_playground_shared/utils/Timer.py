import time
from .GlobalTimeSpentInFunction import GlobalTimeSpentInFunction

class Timer(object):
    def __init__(self, name) -> None:
        self.time = time.time()
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        store = GlobalTimeSpentInFunction()

        delta = time.time() - self.time
        store.save(
            self.name,
            delta
        )
     #   print(f"{self.name} : {delta}")
        self.time = None

if __name__ == "__main__":
    with Timer("name"):
        print("test")
