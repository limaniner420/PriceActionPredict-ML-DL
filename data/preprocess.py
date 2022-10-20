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
