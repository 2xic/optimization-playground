import time


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GlobalTimeSpentInFunction(metaclass=Singleton):
    def __init__(self) -> None:
        self.timers = {}

    def save(self, name, value):
        self.timers[name] = self.timers.get(name, 0) + value

    def hook(self, function):
        original = function

        def hook(*args):
            start = time.time()
            results = original(*args)
            end = time.time()
            self.save(
                function.__name__,
                end - start 
            )
            return results
        return hook


class TestClass:
    def test(self, a, b, c):
        time.sleep(2)
        print((a, b, c))


if __name__ == "__main__":
    obj = TestClass()

    x = GlobalTimeSpentInFunction()
    obj.test = x.hook(obj.test)

    obj.test(1, 2, 3)
    print(x.timers)

    obj.test(1, 2, 3)
    print(x.timers)
