import pandas as pd

SET_MIN = ["date", "close" "symbol", "volume", "change", "changePercent"]
SET_ESS = ["date", "close", "high", "low", "open", "symbol", "volume", "change", "changePercent"]

def read_local(path: str, mode: str = "essential"):
    data = pd.read_json(path)

    if(mode == "minimal"):
        data = data[SET_MIN]
    elif(mode == "essential"):
        data = data[SET_ESS]

    return data