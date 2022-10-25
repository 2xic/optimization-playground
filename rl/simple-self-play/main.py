import torch
from optimization_utils.envs.SimpleEnv import SimpleEnv
from optimization_utils.exploration.EpsilonGreedyManual import EpsilonGreedyManual
import random
from model import SimpleModel
from game import Game
from parameters import GAMES_EACH_EPOCH
from optimization_utils.diagnostics.Diagnostics import Diagnostics

env = SimpleEnv()

def play(model: SimpleModel, epsilon: EpsilonGreedyManual, optimizer: torch.optim.Adam, diagnostics: Diagnostics):
    global env
    env.reset()

    games = []
    for i in range(GAMES_EACH_EPOCH):
        game = Game()
        total_reward = 0
        while not env.done():
            action_distribution = model(env.env.float())
            #print(action_distribution)
            action = epsilon.get_action(lambda: torch.argmax(action_distribution).item())

            (_, reward, action, _) = env.step(action)
            total_reward += reward

            game.add(action_distribution, action)
            diagnostics.profile(action)
            epsilon.update()

    #    print(total_reward)

        game.set_score(total_reward)
        games.append(game)
        diagnostics.reward(total_reward)
        total_reward = 0


    """
    Hm, this simple way does nto seem to convergence.
    """
    games_sorted = sorted(games, key=lambda x: x.score)
    bad_games = games_sorted[:GAMES_EACH_EPOCH // 2]
    good_games = games_sorted[GAMES_EACH_EPOCH // 2:]

    loss = torch.tensor(0).float()
    # actions done in the lost games = bad
    for i in bad_games:
        for j in i.action_distributions:
            loss += (-min(i.score, 1) * j).sum()

    # actions done in the win games = good
    for i in good_games:
        for j in i.action_distributions:
            loss += (i.score * j).sum()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss

if __name__ == "__main__":
    model = SimpleModel(
        input_size=env.state_size,
        output_size=env.action_space,
    )
    epsilon = EpsilonGreedyManual(env.action_space - 1)
    optimizer = torch.optim.Adam(model.parameters())

    diagnostics = Diagnostics()

    for epoch in range(10_000):
        loss = play(model, epsilon, optimizer, diagnostics)

        if epoch % 100 == 0:
            diagnostics.print(epoch, metadata={
                "loss": loss.item() 
            })

