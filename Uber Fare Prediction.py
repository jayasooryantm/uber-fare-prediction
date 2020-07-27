import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv(r'uber data.csv')
testdata = pd.read_csv(r'test.csv')



data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])


data['date'] = data['pickup_datetime'].apply(lambda d: d.day)
data['month'] = data['pickup_datetime'].apply(lambda d: d.month)
data['year'] = data['pickup_datetime'].apply(lambda d: d.year)
data['day'] = data['pickup_datetime'].apply(lambda d: d.weekday)
data['hour'] = data['pickup_datetime'].apply(lambda d: d.hour)


data.drop(['key','pickup_datetime'], axis=1, inplace=True)

data.dropna(how='any',inplace=True)

min_lat = -90
max_lat = 90
min_long = -180
max_long = 180

data.drop(data[(data['pickup_longitude'] < min_long) | (data['pickup_longitude'] > max_long)].index, inplace=True)
data.drop(data[(data['dropoff_longitude'] < min_long) | (data['dropoff_longitude'] > max_long)].index, inplace=True)
data.drop(data[(data['pickup_latitude'] < min_lat) | (data['pickup_latitude'] > max_lat)].index, inplace=True)
data.drop(data[(data['dropoff_latitude'] < min_lat) | (data['dropoff_latitude'] > max_lat)].index, inplace=True)


data.drop(data[data['fare_amount']<=0].index,inplace=True)
data.drop(data[data['passenger_count']==0].index,inplace=True)


x = data.drop('fare_amount',axis=1)
y = data['fare_amount']


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(bootstrap=True, ccp_alpha=0.0,
                                             criterion='mse', max_depth=None,
                                             max_features='auto',
                                             max_leaf_nodes=None,
                                             max_samples=None,
                                             min_impurity_decrease=0.0,
                                             min_impurity_split=None,
                                             min_samples_leaf=1,
                                             min_samples_split=2,
                                             min_weight_fraction_leaf=0.0,
                                             n_estimators=100, n_jobs=None,
                                             oob_score=False, random_state=42,
                                             verbose=0, warm_start=False)


regressor.fit(x_train,y_train)


predict = regressor.predict(x_test)


from sklearn.metrics import r2_score, mean_squared_error



print("RMSE is: ", np.sqrt(mean_squared_error(y_test,predict)))

