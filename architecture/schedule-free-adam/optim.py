from typing import Any, Dict
from torch.optim import Optimizer
import torch
from dataclasses import dataclass

@dataclass
class ParameterState:
    b1 : float
    b2 : float
    warm_up_steps : int
    decay: float
    # Need to be init based on value
    z: torch.Tensor
    x: torch.Tensor
    v: torch.Tensor
    c: torch.Tensor
    t: torch.Tensor
    
class ScheduleFreeOptimizer(Optimizer):
    def __init__(self, params, lr=1e-3) -> None:
        defaults = {
            "lr": lr 
        }
        super().__init__(params, defaults)
        # init the state
        self.state = dict() 
        for group in self.param_groups: 
            for p in group['params']: 
                self.state[p] = self.get_params(p.data)

    def step(self):
        eps = 1e-8
        for group in self.param_groups: 
            for p in group['params']: 
                if p not in self.state: 
                    self.state[p] = self.get_params(p.data)                    
                state = self.state[p]
                lr = group['lr']
                gradient = p.grad.data

                y_t = (1 - state.b1) * state.z + state.b1 * state.x
                state.v = state.b2 * state.v + (1 - state.b2) * (gradient ** 2)                
                v_norm = state.v / (1 - state.b2**state.t) 

                lr = lr * min(1, state.t / state.warm_up_steps)
                state.z = state.z - lr * gradient / (torch.sqrt(v_norm) + eps) - lr * state.decay * y_t
                # todo: optimize this 
                state.c = (lr ** 2) / ((torch.ones(state.t)*lr ** 2).sum())
                state.x = (1 - state.c) * state.x + state.c * state.z
                p.data = state.x
                state.t += 1

    def get_params(self, data):
        return ParameterState(**{
            "b1": 0.9,
            "b2": 0.999,
            "warm_up_steps": 1,
            "decay": 0,
            # Need to be init based on value
            "z": data.clone(),
            "x": data.clone(),
            "v": 0,
            "c": 0,
            "t": 1,
        })
    