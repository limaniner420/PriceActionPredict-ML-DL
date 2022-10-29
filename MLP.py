from sklearn.neural_network import _multilayer_perceptron as mlp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

file = pd.read_json("data.json")
data = pd.DataFrame(file['GOOG']['chart'])
data.set_index('date',inplace = True)
data["log_returns"] = np.log(1 + data['close'].pct_change())
data = data.filter(['close','log_returns']).rename(columns = {"close" : "t"})
# data = data.filter(['changePercent']).rename(columns = {"changePercent" : "t"})
data["t-2"] = data["t"].shift(2)
data["t-1"] = data["t"].shift(1)
data["t+1"] = data["t"].shift(-1)
data = data[["t-2","t-1","t","t+1","log_returns"]]
data.dropna(inplace = True)

# X_train, X_test, y_train, y_test = train_test_split(data[["t","log_returns"]], data["t+1"], test_size=0.25, shuffle=False)
X_train, X_test, y_train, y_test = train_test_split(data[["t-2","t-1","t","log_returns"]], data["t+1"], test_size=0.25, shuffle=False)

scaler = StandardScaler()
pipeline = make_pipeline(scaler, mlp.MLPRegressor(max_iter = 1000, shuffle = False))
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print('r2_score: ',r2_score(y_test, predictions))
print('MAE: ', mean_absolute_error(y_test, predictions))
print('MSE: ', mean_squared_error(y_test, predictions, squared=True))
print("MAPE: ", mean_absolute_percentage_error(y_test, predictions))
print('rmse: ',mean_squared_error(y_test, predictions, squared=False))

plt.figure(figsize=(20,10))
plt.xlabel('Date', fontsize = 16)
plt.ylabel('Closing Price', fontsize = 16)
plt.plot(y_test, linewidth=1)
plt.plot(predictions, linewidth=1)
plt.legend(['Actual', 'Predictions'], loc = 'lower right')
plt.show()