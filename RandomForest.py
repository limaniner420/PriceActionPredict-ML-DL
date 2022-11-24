import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import data.preprocess as pre
import optuna
import data.preprocess as pre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def randomforst_fun(symbol,apply_inds: str="none", window : int=1, optimise: bool = False , opt_params : set = None):
  SET_ESS_none = ["t"]
  SET_ESS_cross = ["t","ma_cross","macd_cross"]
  SET_ESS_quant = ["t","rsi","ma_long","ma_short","volatility", "K","D"]
  SET_ESS_full = ["t","ma_cross","macd_cross","ma_long","macd_cross","K","D", "volatility"]
  if symbol == "SPY":
    file = pd.read_json("SPY.json")
  elif symbol == "COST":
    file = pd.read_json("COST.json")
  elif symbol == "AAPL":
    file = pd.read_json("AAPL.json")
  elif symbol == "GOOG":
    file = pd.read_json("GOOG.json")
    
  
  data = pd.DataFrame(file[symbol]['chart'])
  pre.apply_indicators(data)
  data.set_index('date',inplace = True)
  data = data.drop(["high", "low", "open", "symbol", "volume", "change", "changePercent"], axis = 1).rename(columns =  {"close": "t"})
  if apply_inds == "none":
    data = data.filter(SET_ESS_none)
  elif apply_inds == "cross":
    data = data.filter(SET_ESS_cross)
  elif apply_inds == "quant":
    data = data.filter(SET_ESS_quant)
  elif apply_inds == "full":
    data = data.filter(SET_ESS_full)
    
  for i in range(1, window):
    label = "t-" + str(i)
    data[label] = data["t"].shift(i)
  data["y"] = data["t"].shift(-1)
  data.dropna(inplace = True)
  
  x = data.drop(["y"], axis = 1)
  y = data["y"]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.25, shuffle =False)
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  
  def run(trial):
    space = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500, 100),
        'random_state': trial.suggest_categorical('random_state', [1, 2, 30, 42]),
        'min_samples_split': trial.suggest_categorical('min_samples_split', [2,5,10]),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1,15,2),
        'max_depth': trial.suggest_int('max_depth', 1, 15,1),
        'bootstrap': trial.suggest_categorical('bootstrap', [True,False])
    }
    model = RandomForestRegressor(n_estimators=space['n_estimators'], random_state=space['random_state'], min_samples_split=space['min_samples_split'], min_samples_leaf=space['min_samples_split'], 
                                  max_depth=space['max_depth'], bootstrap=space['bootstrap'])
    model.fit(x_train, y_train)
    predict = model.predict(x_test)
    
    mape = mean_absolute_percentage_error(y_test, predict)
    r2 = r2_score(y_test, predict)
        
    return mape,r2
  
  if(optimise == True):
    if(opt_params == None):
      study = optuna.create_study(directions=["minimize","maximize"])
      study.optimize(run, n_trials=200)
      best_params = min(study.best_trials, key=lambda t: t.values[0]).params
      params = {'n_estimators': best_params["n_estimators"],'random_state': best_params["random_state"], 'min_samples_split': best_params["min_samples_split"], 
                'min_samples_leaf': best_params["min_samples_leaf"],'max_depth': best_params["max_depth"] ,'bootstrap': best_params["bootstrap"]}
    else:
      params = {'n_estimators': opt_params["n_estimators"],'random_state': opt_params["random_state"], 'min_samples_split': opt_params["min_samples_split"], 
            'min_samples_leaf': opt_params["min_samples_leaf"],'max_depth': opt_params["max_depth"],'bootstrap': opt_params["bootstrap"]}
  else:
     params = {'n_estimators': 500,'random_state': 42, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth' : 10, 'bootstrap' : True}
  
  model = RandomForestRegressor(n_estimators=params['n_estimators'], random_state=params['random_state'], min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf']
                                , max_depth=params['max_depth'], bootstrap=params['bootstrap'])
  model.fit(x_train, y_train)
  predict = model.predict(x_test)
  r2 = r2_score(y_test, predict)
  mape = mean_absolute_percentage_error(y_test, predict)
  metric_values = {"r2" : r2,
              "mape": mape, 
              "n_estimators": params["n_estimators"],
              "random_state": params["random_state"],
              "min_samples_split": params["min_samples_split"],
              "min_samples_leaf": params["min_samples_leaf"],
              "max_depth": params["max_depth"],
              "bootstrap": params['bootstrap']
              }
  p = pd.DataFrame(predict)
  p["date"] = y_test.index
  p.set_index("date",inplace = True)
  return metric_values, y_test, p, params
