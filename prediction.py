import pandas
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt


filename = 'prediction.csv' # change if necessary
# import data
df = pandas.read_csv(filename, index_col='Month', parse_dates=True)
df = df[:-1]

model = ARIMA(df['Cases'], order=(1,1,1)).fit() # create model

# prepare predictions dataset
# in the next line, 1 represents Jan 2020, and 28 represents Apr 2022; lesser numbers represent earlier dates and greater numbers represent later dates
pred = pandas.DataFrame(list(model.predict(start=1, end=28, typ='levels')), columns=['Cases'])
pred.index = df.index

# plot both datasets
plt.plot(pred, color='red')
plt.plot(df, color='black')
plt.show()