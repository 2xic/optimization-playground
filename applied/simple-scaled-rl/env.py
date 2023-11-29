from pyboy import PyBoy, WindowEvent
import numpy as np
import torch
import multiprocessing
from pyboy.logger import log_level
from typing import List 
log_level("DISABLE")

"""
Goal of the class is to collect state-action pairs simply
"""

class GameResults:
    def __init__(self, state_action_pairs, score, model_id):
        self.state_action_pairs = state_action_pairs
        self.score = score
        self.model_id = model_id

    def get_state_action_pairs(self):
        returns = []
        gamma = .99
        reward = 0
        for (_, _, _, r) in self.state_action_pairs[::-1]:
            reward = r + gamma * reward
            returns.insert(0, reward)
        
        for index, (prev_state, state, action, _) in enumerate(self.state_action_pairs):
            yield (prev_state, state, action, returns[index]) 

    def __str__(self):
        return "Length: {size}".format(size=len(self.state_action_pairs))
    
    def __repr__(self):
        return self.__str__()

class GamePool:
    def __init__(self, n, debug=False):
        self.size = n
        self.num_processes = multiprocessing.cpu_count()
        self.debug = debug

    def run(self, model) -> List[GameResults]:
        output = []
        try:
            with multiprocessing.Pool(self.num_processes) as p:
                args_list = [
                    (model, 0, run_id) if type(model) != list else (model[run_id % len(model)], run_id % len(model), run_id)
                    for run_id in range(self.size)
                ]
                results = p.map(self._run_game, args_list)
                output += results
        except KeyboardInterrupt:
            p.terminate()
        except Exception as e:
            print(e)
            p.terminate()
        finally:
            p.join()
        return output

    def _run_game(self, args):
        (model, model_id, run_id) = args
        if self.debug:
            print(f"Staring {model_id}")
        py_boy = PyBoy('bins/Tetris.gb', window_type="headless", game_wrapper=True)
        py_boy.set_emulation_speed(0)

        tetris = py_boy.game_wrapper()
        tetris.start_game()

        prev_state = None
        count = 0
        state_action_pairs = []
        actions = {}
        possible_actions = {
            0: [WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT],
            1: [WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT],
            2: [WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN],
            3: [WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A],
            4: [WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B],
        }

        while not tetris.game_over():
            py_boy.tick()
            game_area = np.asarray(tetris.game_area())
            if prev_state is None or np.any((prev_state != game_area)):
                torch_game_area = torch.from_numpy(game_area.astype(np.float32))
                action = model.get_action(torch_game_area)
                actions[action.item()] = True

                for index in list(actions.keys()):
                    if index != action:
                        py_boy.send_input(possible_actions[index][1])
                        del actions[index]
                    else:
                        py_boy.send_input(possible_actions[index][0])
                if prev_state is not None:
                    torch_prev_state = torch.from_numpy(prev_state.astype(np.float32))
                    torch_game_area = torch.from_numpy(game_area.astype(np.float32))
                    state_action_pairs.append(
                        (torch_prev_state, torch_game_area , action, 0)
                    )
                count += 1
                prev_state = game_area
        state_action_pairs.append(
            (torch_prev_state, torch_game_area, action, tetris.score)
        )
        if self.debug:
            print(f"Run id: {run_id}, score: {tetris.score}")
        return GameResults(state_action_pairs, tetris.score, model_id)

if __name__ == "__main__":
    results = GamePool(n=15).run()
    print(results)
