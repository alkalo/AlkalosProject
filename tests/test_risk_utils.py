from risk_utils import position_sizing, daily_kill_switch, apply_fee


def test_position_sizing_basic():
    assert position_sizing(10000) == 10000 * 0.005 / 0.01


def test_apply_fee():
    assert apply_fee(100, 0.1) == 90


def test_daily_kill_switch():
    assert daily_kill_switch([100, 98]) is True
    assert daily_kill_switch([100, 99]) is False
