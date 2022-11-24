import math
from tkinter.tix import X_REGION
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import cross_val_score
import data.preprocess as pre
import sklearn.metrics as metrics
import optuna

SET_ESS = ["date", "close", "high", "low", "open", "symbol", "volume", "change", "changePercent"]

def SVRprice(sym: str, opt_params, apply_inds: str = "none", window: int = 1, optimise: bool = False):
    try:
        file = pd.read_json(sym + ".json")
        data = pd.DataFrame(file[sym]['chart'])[SET_ESS]

    except Exception as e:
        print("Invalid Symbol")
        exit()


    pre.apply_indicators(data)

    if(apply_inds == 'full'):
        data = data[['date','close','ma_cross', 'macd_cross','rsi', 'volatility', 'K', 'D', 'ma_short', 'ma_long']]
    elif(apply_inds == 'cross'): 
        data = data[['date','close', 'ma_cross', 'macd_cross']]
    elif(apply_inds == 'quant'): 
        data = data[['date','close', 'rsi', 'volatility', 'K', 'D', 'ma_short', 'ma_long']]
    elif(apply_inds == 'none'):
         data = data[['date','close']]
    else:
        print("Invalid Indicator Input")
        exit()
    
    data = data.rename(columns = {"close" : "t"})
    for i in range(1, window):
        label = "t-" + str(i)
        data[label] = data["t"].shift(i)

    data["y"] = data["t"].shift(-1)
    #data['prediction'] = data[['close']].rolling(ipt).mean()
    data.dropna(inplace = True) 
    data.set_index("date",inplace = True)

    #X = np.array(data.drop(['prediction'], axis=1))
    X = data.drop(['y'], axis=1)
  
    #y = np.array(data['prediction'])
    y = data['y']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.145, shuffle =False)

    def objective(trial):
        kernel = trial.suggest_categorical('kernel', ['rbf']) #'linear','rbf','poly','sigmoid','precomputed'
        gamma = trial.suggest_float('gamma',0,10)
        C = trial.suggest_float('C',1,1000)
        epsilon = trial.suggest_float('epsilon',0,10)

        # epsilon = width of the tube around the hyperplane. 
        # C = how much we care about the error. Larger number means better fit but risk overfitting.
        # gamma = how much influence a single training example has

        regr = SVR(kernel = kernel, C = C ,epsilon = epsilon, gamma = gamma)
        regr.fit(x_train, y_train)
        predict = regr.predict(x_test)
        r2 = r2_score(predict, y_test)
        mae = mean_absolute_error(predict, y_test)
        mse = mean_squared_error(predict, y_test, squared=True)
        mape = mean_absolute_percentage_error(predict, y_test)
        rmse  = mean_squared_error(predict, y_test, squared=False)

        return r2, mae, mse, mape, rmse

    if(optimise and opt_params == None):
        study = optuna.create_study(directions=["maximize", "minimize", "minimize", "minimize", "minimize"])
        study.optimize(objective, n_trials=5000)
        optimised_SVR = SVR(kernel = study.best_trials[0].params['kernel'] , gamma = study.best_trials[0].params['gamma'], C = study.best_trials[0].params['C'],epsilon = study.best_trials[0].params['epsilon'])
        optimised_SVR.fit(x_train, y_train)
        predict_y = pd.DataFrame(optimised_SVR.predict(x_test))
        predict_y["date"] = x_test.index
        predict_y.set_index("date",inplace = True)

        metric_values = {"r2" : metrics.r2_score(y_test, predict_y),
                    "mape": metrics.mean_absolute_percentage_error(y_test, predict_y), 
                    "kernel": study.best_trials[0].params["kernel"],
                    "gamma": study.best_trials[0].params["gamma"],
                    "C": study.best_trials[0].params["C"],
                    "epsilon": study.best_trials[0].params["epsilon"],
                    }
        opt_params = {"kernel": study.best_trials[0].params["kernel"],
                    "gamma": study.best_trials[0].params["gamma"],
                    "C": study.best_trials[0].params["C"],
                    "epsilon": study.best_trials[0].params["epsilon"],
                    }

    elif(optimise):
        model = SVR(kernel=opt_params['kernel'], gamma=opt_params['gamma'], C = opt_params['C'], epsilon=opt_params['epsilon'])
        model.fit(x_train, y_train)
        predict_y = pd.DataFrame(model.predict(x_test))
        predict_y["date"] = x_test.index
        predict_y.set_index("date",inplace = True)

        metric_values = {"r2" : metrics.r2_score(y_test, predict_y),
                    "mape": metrics.mean_absolute_percentage_error(y_test, predict_y), 
                    "kernel": 'rbf',
                    "gamma": opt_params['gamma'],
                    "C": opt_params['C'],
                    "epsilon": opt_params['epsilon'],
                    }
    else:
        model = SVR(kernel='rbf', C = 1e2, gamma=0.1, epsilon=0.2)
        model.fit(x_train, y_train)
        predict_y = pd.DataFrame(model.predict(x_test))
        predict_y["date"] = x_test.index
        predict_y.set_index("date",inplace = True)

        metric_values = {"r2" : metrics.r2_score(y_test, predict_y),
                    "mape": metrics.mean_absolute_percentage_error(y_test, predict_y), 
                    "kernel": 'rbf',
                    "gamma": 0.1,
                    "C": 100,
                    "epsilon": 0.2,
                    }


    return metric_values, predict_y, y_test, opt_params

if __name__ == "__main__":
    print(SVRprice("GOOG", None, 'none', 1 , 0))
    



