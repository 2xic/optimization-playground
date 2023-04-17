
import unittest
from pbs import PBS
from rebel import Rebel


class TestRebel(unittest.TestCase):
    def test_should_execute(self):
        obj = Rebel(None, None)
        obj.linear(
            PBS()
        )

        assert len(obj.dataset.policy) > 0
        assert len(obj.dataset.value) > 0
        
