from functools import wraps

def decorator(**kwargs_decorator):
    def inner_function(function):
        @wraps(function)
        def wrapper():
            epoch_done = lambda: print("DOne :o")
            # I do call for each search parameter here
            for i in range(0, kwargs_decorator["b"]):
                clone_b = kwargs_decorator
                clone_b["b"] = i
                clone_b["epoch_done"] = epoch_done
                function(**clone_b)
        return wrapper
    return inner_function


@decorator(b=5, a=1)
def train_model(a, b, epoch_done):
    print("a ", a, "b", b)
    epoch_done()
    
if __name__ == "__main__":
    train_model()
