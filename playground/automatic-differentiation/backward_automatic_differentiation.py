"""
Based on 
- https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation
"""
import numpy as np
import math

class Value:
    def __init__(self, value) -> None:
        self.value = float(value)
        self.deriv = None
        self.children = []

    def __mul__(self, other):
        results = Value(
            self.value * other.value
        )
        self._add_results(other, results)
        return results

    def __add__(self, other):
        results = Value(
            self.value + other.value,
        )
        self._add_results(Value(1), results, direct=True)
        other._add_results(Value(1), self, direct=True)
        return results

    def __pow__(self, number):
        results = Value(
            self.value ** number,
        )
        self._add_results(number, results)
        return results

    def _add_results(self, other_value, calculated_value, direct=False):
        self.children.append((other_value.value, calculated_value))
        if not direct:
            other_value.children.append((self.value, calculated_value))

    @staticmethod
    def sin(other):
        results = Value(
            np.sin(other.value)
        )
        other._add_results(Value(math.cos(other.value)), results, direct=True)
        return results

    def grad(self):
        if self.deriv is None:
            self.deriv = sum(
                weight * var.grad()
                for weight, var in self.children
            )
        return self.deriv

if __name__ == "__main__":
    x = Value(0.5)
    y = Value(4.2)

    z = x * y + Value.sin(x)
    z.deriv = 1
    """
            
    """
    assert np.allclose(z.value, 2.579425538604203)
    assert np.allclose(x.grad(), y.value + math.cos(x.value)), y.value + math.cos(x.value)
