import pandas as pd
import numpy as np

def ma_cross(data: pd.DataFrame, t_long: int = 200, t_short: int = 50) -> pd.DataFrame:
    """ 
    data: pandas Dataframe containing historical price data. Requires columns "date", "close". 
    t: Time periods used to calculate moving averages. T_long must be greater than t_short for valid data.
    
    Returns array of integers reflecting moving average crossover/crossunder.

    See: https://www.investopedia.com/terms/c/crossover.asp
    """
    ma = pd.DataFrame()
    ma["ma_long"] = data["close"].rolling(window = t_long).mean()
    ma["ma_short"] = data["close"].rolling(window = t_short).mean()
    ma["current"] = ma["ma_short"] > ma["ma_long"]
    ma["previous"] = ma["current"].shift(1)
    ma.dropna(inplace = True)
    ma["ma_cross"] = np.zeros(ma.shape[0])
    ma["ma_cross"].mask((ma["previous"] == False) & (ma["current"] == True), 1, inplace = True)
    ma["ma_cross"].mask((ma["previous"] == True) & (ma["current"] == False), -1, inplace = True)
    ma = ma.iloc[1:, :]
    return ma["ma_cross"]

def macd_cross(data: pd.DataFrame, t_macd: int = 9, t_fast = 12, t_slow = 26) -> pd.DataFrame:
    """ 
    Uses historical price to calculate MACD indicator.
    data: pandas Dataframe containing historical price data. Requires columns "date", "close". 
    t_macd: Time periods used to calculate MACD trigger line.
    t_fast: Time periods used to calculate fast exponential-weighted average.
    t_slow: Time periods used to calculate slow exponential-weighted average.

    Returns array of integers reflecting macd crossover/crossunder.

    See: https://www.investopedia.com/terms/m/macd.asp
    """
    macd = pd.DataFrame()
    macd["date"] = data["date"]
    macd["MACD"] = data["close"].ewm(span = t_fast, adjust = False, min_periods = t_fast).mean() - data["close"].ewm(span = t_slow, adjust = False, min_periods = t_slow).mean()
    macd["trigger"] = macd["MACD"].ewm(span = t_macd, adjust = False, min_periods = t_macd).mean()
    macd["current"] = macd["trigger"] > macd["MACD"]
    macd["previous"] = macd["current"].shift(1)
    macd.dropna(inplace = True)
    macd["macd_cross"] = np.zeros(macd.shape[0])
    macd["macd_cross"].mask((macd["previous"] == False) & (macd["current"] == True), 1, inplace = True)
    macd["macd_cross"].mask((macd["previous"] == True) & (macd["current"] == False), -1, inplace = True)
    macd = macd.iloc[1:, :]
    return macd["macd_cross"]