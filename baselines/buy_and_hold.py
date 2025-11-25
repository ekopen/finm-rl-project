def buy_and_hold(prices):
    """
    Minimal buy & hold baseline.
    Returns final PnL.
    """
    return prices[-1] / prices[0] - 1.0
