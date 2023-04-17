import torch
from discriminator import Discriminator
from soft_actor_critic.model import SAC
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.replay_buffers.ReplayBuffer import ReplayBuffer
import torch.nn as nn
from optimization_utils.utils.StopTimer import StopTimer
"""
Diversity is all you need
"""
class DIAY:
    def __init__(self, state_size, action_size, skills) -> None:
        # from paper : We fix p(z) to be uniform in our approach, guaranteeing that is has maximum entropy.
        self.probs = [1/skills for _ in range(skills)]
        self.p_z = torch.distributions.Categorical(torch.tensor(self.probs))
        self.step_size = 100
        self.skills = skills
        self.discriminator = Discriminator(state_size, skills)
        self.discriminator_loss = torch.optim.Adam(self.discriminator.parameters())
        self.sac = SAC(state_size, action_size, skills)

    def test_skills(self, env: SimpleEnv):
        for skill in range(self.skills):
            env.reset()
            while not env.done():
                state = env.state.reshape((1, -1))
                condition = self.sac.action_network.get_condition(1, skill)
                (action, _) = self.sac.action(state, condition)
                
                (next_state, _, _, _) = env.step(action)

                print(f"skill : {skill}", state, action, next_state)

    def step(self, env: SimpleEnv):
        replay_buffer = ReplayBuffer()
        while not env.done():
            skill = self.p_z.sample()

            state = env.state.reshape((1, -1))
            condition = self.sac.action_network.get_condition(1, skill)
            (action, _) = self.sac.action(state, condition)
            
            (next_state, _, _, _) = env.step(action)
            next_state = next_state.reshape((1, -1)).float()
            pseudo_reward = self.psuedo_reward(next_state, skill)

            replay_buffer.push(
                state=state,
                reward=pseudo_reward.item(),
                action=action,
                is_last_state=False, #env.done(),
                metadata={
                    "next_state": next_state.clone(),
                    "conditional": condition,
                },
                id=-1
            )

            loss_discriminator = nn.CrossEntropyLoss()(
                self.discriminator(next_state),
                torch.tensor([skill])
            )
            loss_discriminator.backward()

        loss_sac = self.sac.optimize(replay_buffer)
        self.discriminator_loss.step()
        self.discriminator.zero_grad()
#        print(loss)
        return (
            loss_discriminator,
            loss_sac
        )

    def psuedo_reward(self, state, skill):
        return torch.log(self.discriminator(state)[0][skill]) - torch.log(torch.tensor(self.probs[skill]))

if __name__ == "__main__":
    model = DIAY(2, 2, 2)
#    for i in range(10_000):
    timer = StopTimer(
       iterations=10_000,
       timeout_seconds=5*60
    )
    timer.tick()
    while not timer.is_done():
        loss_discriminator, loss_sac = model.step(SimpleEnv())

        if timer.epoch % 100 == 0:
            print(timer.epoch)
            print(loss_discriminator)
            print(loss_sac)
          #  break
        timer.tick()

    torch.save({
        'model_state_dict': model.sac.state_dict(),
    }, "diayn_model")
    
