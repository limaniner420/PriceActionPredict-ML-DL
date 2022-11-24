import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.losses import MeanSquaredError
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, Bidirectional
from tensorflow import keras
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import data.preprocess as pre
import wandb
import math
import keras.callbacks as call
import optuna
import data.preprocess as pre
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def LSTM_function(symbol,apply_inds: str="none", window : int=1, optimise: bool = False , opt_params : set = None):
  SET_ESS_none = ["close"]
  SET_ESS_cross = ["close","ma_cross","macd_cross"]
  SET_ESS_quant = ["close","rsi","ma_long","ma_short","volatility", "K","D"]
  SET_ESS_full = ["close","ma_cross","macd_cross","ma_long","macd_cross","K","D", "volatility"]
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
  if apply_inds == "none":
    data = data.filter(SET_ESS_none)
    feature = 1
  elif apply_inds == "cross":
    data = data.filter(SET_ESS_cross)
    feature = 3
  elif apply_inds == "quant":
    data = data.filter(SET_ESS_quant)
    feature = 7
  elif apply_inds == "full":
    data = data.filter(SET_ESS_full)
    feature = 8
  
  data.dropna(inplace=True)
  data_len = len(data.values)

  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data.values)
  # predict 360 days
  test_size = math.ceil(0.25*data_len)
  # 14 days to predict the next day (sliding window)
  train_data = scaled_data[0:data_len-test_size-window,:]
  x_train = []
  y_train = []
  for i in range(window, len(train_data)): 
    x_train.append(train_data[i-window:i, :]) 
    y_train.append(train_data[i][0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], feature))

  #split the test_data (for validation)
  test_data = scaled_data[data_len-(window+test_size):data_len,:]

  x_test = []
  y_test = []

  for i in range(window, len(test_data)): #run from 14, 15 , 16 ... len(test_data)-1
    x_test.append(test_data[i-window:i, :])
    y_test.append(test_data[i][0])
    
  x_test,y_test = np.array(x_test), np.array(y_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], feature))
  
  def run(trial):
        space = {
            'units': trial.suggest_int('units', 150, 250, 50),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2,0.6),
            #'dropout_rate': trial.suggest_categorical('dropout_rate', [0.2, 0.4,0.6]),
            'epochs': trial.suggest_int('epochs', 150, 250, 50),
            #'learning_rate': trial.suggest_categorical('learning_rate', [0.002,0.004,0.006])
            'learning_rate': trial.suggest_float('learning_rate', 0.002,0.006)
        }
        
        callback = call.EarlyStopping(monitor='val_loss', patience=10)
        model = Sequential()
        model.add(LSTM(units=space['units'], return_sequences = True, input_shape=(x_train.shape[1], feature)))
        model.add(Dropout(rate=space['dropout_rate']))
        model.add(LSTM(units=space['units']))
        model.add(Dropout(rate=space['dropout_rate']))
        model.add(Dense(units=14))
        model.add(Dense(units=1))
            
        model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Nadam(learning_rate=space['learning_rate']), metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=space['epochs'], validation_data=(x_test, y_test), batch_size=256, verbose=0, callbacks = [callback])
        
        predictions = model.predict(x_test)
        Predict_dataset_3D = np.zeros(shape=(len(predictions), feature))
        Predict_dataset_3D[:,0] = predictions[:,0]
        predictions = scaler.inverse_transform(Predict_dataset_3D)[:,0]
        predict = data[data_len-test_size:data_len]
        mape = mean_absolute_percentage_error(predict['close'].values, predictions)
        r2 = r2_score(predict['close'].values, predictions)
        return mape,r2
  
  
  if(optimise == True):
    if(opt_params == None):
      study = optuna.create_study(directions=["minimize","maximize"])
      study.optimize(run, n_trials=15)
      best_params = min(study.best_trials, key=lambda t: t.values[0]).params
      params = {'units': best_params["units"],'dropout_rate': best_params["dropout_rate"], 'epochs': best_params["epochs"], 'learning_rate': best_params["learning_rate"]}
    else:
      params = {'units': opt_params["units"],'dropout_rate':opt_params["dropout_rate"], 'epochs': opt_params["epochs"], 'learning_rate':opt_params["learning_rate"]}
  else:
     params = {'units': 200,'dropout_rate': 0.4, 'epochs': 200, 'learning_rate': 0.006}
  

  #predict future +1 day value

  callback = call.EarlyStopping(monitor='val_loss', patience=10)
  model= Sequential()
  model.add(LSTM(units=params['units'], return_sequences=True, input_shape=(x_train.shape[1],feature)))
  model.add(Dropout(rate=params['dropout_rate']))

  model.add(LSTM(units=params['units'], input_shape=(x_train.shape[1],feature)))
  model.add(Dropout(rate=params['dropout_rate']))
  model.add(Dense(units=14))
  model.add(Dense(units=1))

  model.summary()
  model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Nadam(learning_rate= 0.006907009401849758))
  model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=params['epochs'], batch_size=256)#callbacks=[WandbCallback()  
  predictions = model.predict(x_test)
  Predict_dataset_3D = np.zeros(shape=(len(predictions), feature) )
  Predict_dataset_3D[:,0] = predictions[:,0]
  
  predictions = scaler.inverse_transform(Predict_dataset_3D)[:,0]
  predict = data[data_len-test_size:data_len]
  predict['Predictions'] = predictions
  
  mape= mean_absolute_percentage_error(predict['close'].values, predictions)
  r2 = r2_score(predict['close'].values, predictions)
  metric_values = {"r2" : r2,
                "mape": mape, 
                "units": params["units"],
                "dropout_rate": params["dropout_rate"],
                "epochs": params["epochs"],
                "learning_rate": params["learning_rate"],
                }
  return metric_values, predict['Predictions'], predict['close'], params
