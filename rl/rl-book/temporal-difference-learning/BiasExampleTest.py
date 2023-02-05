from BiasExample import BiasExample

def test_bias_example():
    env = BiasExample()
    assert env.play(0) == 0
    assert env.done == True

    env.reset()
    assert env.play(1) == 0
    assert env.done == False
    assert env.play(1) != 0
    assert env.done == True
    env.reset()

    assert env.play(0) == 0
    assert env.done == True
