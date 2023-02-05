from WeightSchedule import WeighSchedule

def test_weight_schedule():
    obj = WeighSchedule(None, None, 100)
    t = obj.update_t(0).item()
    assert (1 - t) < 0.005
    t = obj.update_t(1).item()
    assert 0 < t and t < 1
    t = obj.update_t(94).item()
    assert 0 < t and t < 1
    t = obj.update_t(100).item()
    assert t == 0
