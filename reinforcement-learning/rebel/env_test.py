
import unittest
from env import Gesture, RockPaperScissor, Actions


class TestEnv(unittest.TestCase):
    def test_action(self):
        assert type(RockPaperScissor().play(1)) == int

    def test_actions(self):
        assert Gesture(
            Actions.PAPER.value
        ).winner(
            Gesture(Actions.ROCK.value)
        ) == 1

        assert Gesture(
            Actions.ROCK.value
        ).winner(
            Gesture(Actions.PAPER.value)
        ) == -1

        assert Gesture(
            Actions.PAPER.value
        ).winner(
            Gesture(Actions.PAPER.value)
        ) == 0

        assert Gesture(
            Actions.SCISSOR.value
        ).winner(
            Gesture(Actions.PAPER.value)
        ) == 2

        assert Gesture(
            Actions.PAPER.value
        ).winner(
            Gesture(Actions.SCISSOR.value)
        ) == -2

