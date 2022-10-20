import pandas as pd

SET_MIN = ["close" "symbol", "volume", "label", "change", "changePercent"]
SET_ESS = ["close", "high", "low", "open", "symbol", "volume", "label", "change", "changePercent"]

def read_local(path: str, mode: str = "essential"):
    data = pd.read_json(path)

    if(mode == "minimal"):
        data = data[SET_MIN]
    elif(mode == "essential"):
        data = data[SET_ESS]

    return data

# def read_remote()
# 'https://cloud.iexapis.com/stable/stock/SPY/chart/1y?token=pk_8ae6e7e04e0241348774a921c73e7629'