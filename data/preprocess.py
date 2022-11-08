import pandas as pd
from . import indicators as indc

SET_MIN = ["date", "close" "symbol", "volume", "change", "changePercent"]
SET_ESS = ["date", "close", "high", "low", "open", "symbol", "volume", "change", "changePercent"]

def read_local(path: str, mode: str = "essential"):
    data = pd.read_json(path)

    if(mode == "minimal"):
        data = data[SET_MIN]
    elif(mode == "essential"):
        data = data[SET_ESS]

    return data

def apply_indicators(data: pd.DataFrame):
    data["ma_cross"] = indc.ma_cross(data, t_long = 20, t_short = 5)
    data["macd_cross"] = indc.macd_cross(data)
    data['rsi'] = indc.rsi(data)['RSI']
    data['volatility'] = indc.volatility(data)
    data[['K', 'D']] = indc.Stochastic(data)[['K', 'D']]
    data[['ma_long', 'ma_short']] = indc.movingAverage(data)[['ma_long', 'ma_short']]

    return data
