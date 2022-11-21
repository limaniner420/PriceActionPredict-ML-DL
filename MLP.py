from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import data.indicators as indc
import data.preprocess as pre
import joblib
import optuna
import sys

SET_ESS = ["date", "close", "high", "low", "open", "symbol", "volume", "change", "changePercent"]

def MLP(symbol: str, apply_inds: str = "none", window: int = 1, optimise: bool = False):
    file = pd.read_json(symbol + ".json")
    data = pd.DataFrame(file[symbol]['chart'])[SET_ESS]

    if(apply_inds != "none"):
        if(apply_inds == "cross"):
            data["ma_cross"] = indc.ma_cross(data, t_long = 20, t_short = 5)
            data["macd_cross"] = indc.macd_cross(data)
        elif(apply_inds == "quant"):
            data = pre.apply_indicators(data)
            data.drop(["ma_cross", "macd_cross"], axis = 1)
        elif(apply_inds == "full"):
            data = pre.apply_indicators(data)
        else: raise Exception("Invalid indicator scope")
        
    data = data.drop(["high", "low", "open", "symbol", "volume", "change", "changePercent"], axis = 1).rename(columns = {"close" : "t"})
    for i in range(1, window):
        label = "t-" + str(i)
        data[label] = data["t"].shift(i)
    data["y"] = data["t"].shift(-1)

    data.dropna(inplace = True)
    data.set_index("date",inplace = True)
    X = data.drop(["y"], axis = 1)
    y = data["y"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)
    if(optimise):
        params = optimise_hp(data)
        pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(hidden_layer_sizes = params["hidden_layer_sizes"], 
                                                                    activation = params["activation"], 
                                                                    solver = params["solver"],
                                                                    alpha = params["alpha"],
                                                                    max_iter = 5000, shuffle = False))
    else:
        params = {'hidden_layer_sizes': 100, 'activation': 'relu', 'solver': 'adam', 'alpha': 0.0001}
        pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(max_iter = 5000, shuffle = False))
    
    pipeline.fit(X_train, y_train)
    p = pd.DataFrame(pipeline.predict(X_test))
    p["date"] = X_test.index
    p.set_index("date",inplace = True)

    metric_values = {"r2" : metrics.r2_score(y_test, p),
                    "mape": metrics.mean_absolute_percentage_error(y_test, p), 
                    "hidden_layer_sizes": params["hidden_layer_sizes"],
                    "activation": params["activation"],
                    "solver": params["solver"],
                    "alpha": params["alpha"],
                    }
    return metric_values, p, y_test

def optimise_hp(data: pd.DataFrame):
    
    X = data.drop(["t+1"], axis = 1)
    y = data["t+1"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

    def objective(trial):
        space = {
            "hidden_layer_sizes": trial.suggest_int('hidden_layer_sizes', 50, 200, 50),
            "activation": trial.suggest_categorical("activation", ["logistic", "tanh", "relu"]),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.0100)
        }

        try:
            pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(hidden_layer_sizes = space["hidden_layer_sizes"], 
                                                                        activation = space["activation"], 
                                                                        solver = space["solver"],
                                                                        alpha = space["alpha"],
                                                                        max_iter = 1500, shuffle = False))
            pipeline.fit(X_train, y_train)
            
            p = pipeline.predict(X_test)
            r2 = metrics.r2_score(y_test, p)
            mae = metrics.mean_absolute_error(y_test, p)
            mse = metrics.mean_squared_error(y_test, p, squared=True)
            mape = metrics.mean_absolute_percentage_error(y_test, p)
            rmse = metrics.mean_squared_error(y_test, p, squared=False)

            return r2, mae, mse, mape, rmse
        except Exception as e:
            print(e)
            return sys.float_info.min, sys.float_info.max, sys.float_info.max, sys.float_info.max, sys.float_info.max
    study = optuna.create_study(directions=["maximize", "minimize", "minimize", "minimize", "minimize"])
    study.optimize(objective, n_trials=30)
    result = study.best_params
    return result

# file = pd.read_json("GOOG.json")
# data = pd.DataFrame(file['GOOG']['chart'])[SET_ESS]
# # data["ma_cross"] = indc.ma_cross(data, t_long = 20, t_short = 5)
# # data["macd_cross"] = indc.macd_cross(data)
# data = pre.apply_indicators(data)
# data = data.drop(["high", "low", "open", "symbol", "volume", "change", "changePercent"], axis = 1).rename(columns = {"close" : "t"})
# for i in range(1, 7):
#     label = "t-" + str(i)
#     data[label] = data["t"].shift(i)
# data["y"] = data["t"].shift(-1)

# data.dropna(inplace = True)
# data.set_index("date",inplace = True)
# X = data.drop(["y"], axis = 1)
# y = data["y"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, shuffle = False)

# params = pd.DataFrame(pd.read_csv("mlp_trials.csv"))
# params = params.sort_values("values3", axis = 0)

# pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(max_iter = 1500, shuffle = False))
# # pipeline = make_pipeline(StandardScaler(), mlp.MLPRegressor(hidden_layer_sizes = params.iloc[0]["params_hidden_layer_sizes"], 
# #                                                                     activation = params.iloc[0]["params_activation"], 
# #                                                                     solver = params.iloc[0]["params_solver"],
# #                                                                     alpha = params.iloc[0]["params_alpha"],
# #                                                                     max_iter = 5000, shuffle = False))
# pipeline.fit(X_train, y_train)
# p = pd.DataFrame(pipeline.predict(X_test))
# p["date"] = X_test.index
# p.set_index("date",inplace = True)


# metric_values = {"r2" : metrics.r2_score(y_test, p),
#                  "mae": metrics.mean_absolute_error(y_test, p), 
#                  "mse": metrics.mean_squared_error(y_test, p, squared=True), 
#                  "mape": metrics.mean_absolute_percentage_error(y_test, p), 
#                  "rmse": metrics.mean_squared_error(y_test, p, squared=False)}
                 

# print('r2_score: ', metrics.r2_score(y_test, p))
# print('MAE: ', metrics.mean_absolute_error(y_test, p))
# print('MSE: ', metrics.mean_squared_error(y_test, p, squared=True))
# print("MAPE: ", metrics.mean_absolute_percentage_error(y_test, p))
# print('RMSE: ', metrics.mean_squared_error(y_test, p, squared=False))


# plt.figure(figsize=(20, 10))
# plt.xlabel('Date', fontsize = 16)
# plt.ylabel('Close', fontsize = 16)
# plt.xticks(np.arange(0, len(data.values)+1,37),rotation=70)
# plt.plot(y_test, linewidth=1)
# plt.plot(p, linewidth=1)
# plt.legend(['Actual', 'Predictions'], loc = 'lower right')
# plt.savefig("predictions")
# # plt.show()


