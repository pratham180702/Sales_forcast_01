import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

df = pd.read_csv('train.csv', index_col = "Day")
# df = df.asfreq("D")

model = ExponentialSmoothing(endog = df.GrocerySales, trend='add', seasonal='add', seasonal_periods = 7).fit()

predictions = model.forecast(steps =  31)
df['GrocerySales'].plot(figsize= (10,7))
predictions.plot()