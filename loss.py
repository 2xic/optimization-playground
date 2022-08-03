import torch
from model import Model
from prediction import Prediction


class Loss:
    def __init__(self) -> None:
        self.trajectory = []
        self.iterations = 0

    def store_trajectory(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, gamma: torch.Tensor):
        self.trajectory.append(
            (state, action, reward, gamma)
        )

    def iterate(self, model: Model):
        self.K = 3
        for index in range(len(self.trajectory) - self.K):
            model_state = None
            loss = torch.zeros(1)
            Reward = torch.zeros(1)
            for k in range(self.K):                
                (state, action, reward, gamma) = self.trajectory[index + k]

                assert reward.shape == (1, )
                assert gamma.shape == (1, )

                if k == 0:
                    model_state = model.encode(state)
                    Reward += reward
                else:
                    model_state = model.transition(model_state, action)

                model_value = model.value(model_state)
                model_reward, model_gamma = model.outcome(model_state, action)

                loss += self.loss(
                    Prediction(
                        Reward,
                        model_value
                    ),
                    Prediction(
                        model_reward,
                        reward
                    ),
                    Prediction(
                        self._log_self_base(model_gamma),
                        self._log_self_base(gamma)
                    )
                )
                self.iterations += 1
            

    def loss(self, value: Prediction, reward: Prediction, gamma: Prediction):
        return (
            value.delta() + reward.delta() # + gamma.delta()
        )

    def _log_self_base(self, y):
        return torch.log(y) / torch.log(y)
