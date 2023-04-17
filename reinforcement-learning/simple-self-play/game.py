

class Game:
    def __init__(self):
        self.action_distributions = []
        self.score = None

    def add(self, actions, action):
#        self.action_distributions.append(actions)
        self.action_distributions.append(actions[action])

    def set_score(self, score):
        self.score = score

    