from alkalos.strategy import threshold_strategy

def test_thresholds_change_signals():
    indicator = [0.5, 0.7, 0.2]
    signals1 = threshold_strategy(indicator, buy_threshold=0.6, sell_threshold=0.4)
    signals2 = threshold_strategy(indicator, buy_threshold=0.8, sell_threshold=0.1)
    assert signals1 != signals2
