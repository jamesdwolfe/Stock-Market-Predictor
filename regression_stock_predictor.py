import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import yfinance as yf

desired_width=1000
pd.set_option('display.width',desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',100)

stock_name='SPY'
start_date = '2019-06-01'

data=yf.download(stock_name,start_date)
df = data[['Open','Close','High','Low','Volume']]

df['HL_PCT'] = (df['High']-df['Low'])/df['Low'] * 100
df['PCT_change'] = (df['Close']-df['Open'])/df['Open'] * 100

forecast_col = 'Close'
forecast_out=1
df.fillna(-99999,inplace=True)
df['label']=df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

clf.fit(X[:-forecast_out],y[:-forecast_out])
y_new = clf.predict(X_lately)

print(data)
print("Days in advance =", forecast_out)
print("Accuracy percentage =", accuracy)
print("Today's stock price for",stock_name,"=",data['Close'][-forecast_out])
print("Future stock price for",stock_name,"=",y_new[0])
percent_change = (y_new[0]-data['Close'][-forecast_out])/data['Close'][-forecast_out]*100
print("Percentage change =",percent_change)

data.Close.plot()
df.High.plot()
df.Low.plot()
plt.show()