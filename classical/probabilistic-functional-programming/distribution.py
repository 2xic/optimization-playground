from typing import List
from collections import defaultdict
from helpers import distribution_from_outcomes

class Value:
    def __init__(self, value, probability) -> None:
        self.value = value
        self.probability = probability

    def __repr__(self) -> str:
        return f"{self.value} {round(self.probability * 100, 1)}%"

    def __str__(self) -> str:
        return self.__repr__()


class Distribution:
    def __init__(self, outcomes: List[Value]) -> None:
        self.outcomes = outcomes
        self.value_p = {
            value.value:value.probability
            for value in self.outcomes
        }
        assert (sum(list(map(lambda x: x.probability, outcomes))) - 1) < 1e-6

    def __repr__(self):
        return "\n".join(list(map(str, self.outcomes)))

    def __str__(self):
        return self.__repr__()

class BinomialDistribution(Distribution):
    def __init__(self, outcomes: List[float]) -> None:
        assert len(outcomes) == 2
        combined = list(zip([True, False], outcomes))
        super().__init__([
            Value(i, prob)
            for (i, prob) in combined
        ])

    def outcomes_after_n(self, n):
        outcomes = defaultdict(int)
        stack = defaultdict(list)
        entry = range(n)
        for i in entry:
            for a in self.outcomes:
                # First event is 50 % 50
                # Then second is 25 % 
                # etc etc
                for j in stack.get(i - 1, [()]):
                    outcome = j + (a.value, )
                    stack[i].append(outcome)
        """
        This could be simplified
        """
        distributions = []
        for index in entry:
            combined_outcome = defaultdict(int)
            for i in stack[index]:
                combined_outcome[tuple(sorted(i))] += 1
            normalized_p = {}
            for outcomes in combined_outcome:
                sum_p = 1
                for i in outcomes:
                    sum_p *= self.value_p[i]
                normalized_p[outcomes] = sum_p * combined_outcome[outcomes]
            """
            Which we now convert back again
            """
            distribution = Distribution(
                [
                    Value(i, p)
                    for i, p in normalized_p.items()
                ]
            )
            distributions.append(distribution)
        return distributions

class UniformDistribution(Distribution):
    def __init__(self, outcomes):
        uniform = 1 / len(outcomes)
        super().__init__([
            Value(i, uniform)
            for i in outcomes
        ])
 
    def prod(self, other_distribution):
        outcomes = defaultdict(int)
        total = 0
        for a in self.outcomes:
            for b in other_distribution.outcomes:
                results = a.value + b.value
                outcomes[results] += 1
                total += 1

        return Distribution(
            [
                Value(i, count / total)
                for i, count in outcomes.items()
            ]
        )

    def join(self, other_distribution):
        outcomes = defaultdict(int)
        for a in self.outcomes:
            for b in other_distribution.outcomes:
                results = (a.value, b.value)
                outcomes[results] += 1
        return distribution_from_outcomes(outcomes)

