
def distribution_from_outcomes(dict_of_outcomes):
    from distribution import Distribution, Value
    from collections import defaultdict

    total = 0
    outcomes = defaultdict(int)
    for outcome in dict_of_outcomes:
        outcomes[outcome] += dict_of_outcomes[outcome]
        total += dict_of_outcomes[outcome]

    return Distribution(
        [
            Value(i, count / total)
            for i, count in outcomes.items()
        ]
    )
