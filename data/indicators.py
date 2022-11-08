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


def rsi(data: pd.DataFrame, t_periods: int = 14):
    """
    Uses historical price to calcuate the relative strength index
    data: pandas Dataframe containing historical price data. Requires columns "date", "close".
    periods: required time to calculate the rsi
    Returns a Dataframe with the relative strength index and corresponding date
    """
    delta = data['close'].diff()
    rsi = pd.DataFrame()
    rsi['date'] = data['date']
    rsi['gain'] = delta.clip(lower=0) 
    rsi['loss'] = -1*delta.clip(upper=0)
    gain_ema = rsi['gain'].ewm(com=t_periods-1, adjust=False,min_periods=t_periods).mean()
    loss_ema = rsi['loss'].ewm(com=t_periods-1, adjust=False,min_periods=t_periods).mean()
    rs = gain_ema/loss_ema

    rsi['RSI'] = 100 - (100/(1 + rs))
    #rsi = rsi.set_index("date").dropna()
    rsi.dropna(inplace = True)
    return rsi

def volatility(data: pd.DataFrame):
    """"
    Uses historical price to calcuate volatility
    data: pandas Dataframe containing historical price data. Requires columns "date", "close".
    Returns a volatility value during the corresponding period
    """
    data = np.log(data['close']/data['close'].shift(1))
    data = data.fillna(0)
    volatility = data.rolling(window=len(data)).std()*np.sqrt(len(data))
    return volatility.iloc[len(data)-1]

def Stochastic(data: pd.DataFrame):
    """
    Uses historical price to calculate %K and %D
    data: pandas Dataframe containing historical price data. Requires columns "date", "close".
    https://www.investopedia.com/terms/s/stochasticoscillator.asp
    """
    Oscillator = pd.DataFrame()
    Oscillator['date'] = data['date']
    Oscillator['max'] = data['high'].rolling(14).max()
    Oscillator['min'] = data['low'].rolling(14).min()
    Oscillator['K'] = (data['close']-Oscillator['min'])*100/(Oscillator['max'] - Oscillator['min'])
    Oscillator['D'] = Oscillator['K'].rolling(3).mean()
    # Oscillator = Oscillator.set_index("date").dropna()
    Oscillator.dropna(inplace=True)
    return Oscillator

def movingAverage(data: pd.DataFrame, t_long: int = 200, t_short: int = 50) -> pd.DataFrame:
    """ 
    data: pandas Dataframe containing historical price data. Requires columns "date", "close". 
    t: Time periods used to calculate moving averages. T_long must be greater than t_short for valid data.
    
    Returns Dataframe containing fast and slow moving averages.
    See: https://www.investopedia.com/terms/c/crossover.asp
    """

    # TODO: validity check.
    ma = pd.DataFrame()
    ma["date"] = data["date"]
    ma["ma_long"] = data["close"].rolling(window = t_long).mean()
    ma["ma_short"] = data["close"].rolling(window = t_short).mean()
    # ma = ma.set_index("date").dropna()
    ma.dropna(inplace=True)
    return ma

