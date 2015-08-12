# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:33:20 2015

@author: akommajesula
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# importing the metro ridership data downloaded through DC Metrorail
rider = pd.read_csv('ridership.csv')
rider.columns.values
rider.head()
rider.describe()
rider['Date'] = pd.to_datetime(rider['Date'])

#importing weather data. This data only comes in a year so each year has to be downloaded by 
weather2005 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2005/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2005&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2005=weather2005.rename(columns={'EST':'Date'})

weather2006 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2006/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2006&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2006 = weather2006.rename(columns={'EST':'Date'})

weather2007 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2007/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2007&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2007 = weather2007.rename(columns={'EST':'Date'})

weather2008 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2008/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2008&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2008 = weather2008.rename(columns={'EST':'Date'})

weather2009 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2009/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2009&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2009 = weather2009.rename(columns={'EST':'Date'})

weather2010 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2010/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2010&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2010 = weather2010.rename(columns={'EST':'Date'})

weather2011 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2011/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2011&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2011 = weather2011.rename(columns={'EST':'Date'})

weather2012 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2012/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2012&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2012 = weather2012.rename(columns={'EST':'Date'})

weather2013 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2013/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2013&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2013 = weather2013.rename(columns={'EST':'Date'})

weather2014 = pd.read_csv('http://www.wunderground.com/history/airport/KDCA/2014/1/1/CustomHistory.html?dayend=31&monthend=12&yearend=2014&req_city=&req_state=&req_statename=&reqdb.zip=&reqdb.magic=&reqdb.wmo=&format=1')
weather2014 = weather2014.rename(columns={'EST':'Date'})

frames = [weather2005, weather2006, weather2007, weather2008, weather2009, weather2010, weather2011, weather2012, weather2013, weather2014]
#Concatenating all weather data. 
weather = pd.concat(frames)
weather['Date'] = pd.to_datetime(weather['Date'])


weather=weather.drop([' CloudCover',' Max Gust SpeedMPH',' Max Sea Level PressureIn',' Max VisibilityMiles',' Max Wind SpeedMPH',' Mean Humidity',' Mean Sea Level PressureIn',' Mean VisibilityMiles',' Mean Wind SpeedMPH',' Min Humidity',
       ' Min Sea Level PressureIn', ' Min VisibilityMiles', ' WindDirDegrees<br />', 'Max Humidity','Max TemperatureF','MeanDew PointF', 'Min DewpointF', 'Min TemperatureF','Max Dew PointF'], axis=1)    
# Gas prices were available for download as a csv file. 

gas = pd.read_csv('gas2.csv')
gas.columns.values
gas.head()
gas.describe()
gas['Date'] = pd.to_datetime(gas['Date'])

#Uploadiing smartrip data
smartrip = pd.read_csv('smartrip.csv')
smartrip['Date'] = pd.to_datetime(smartrip['Date'])

#Merging the data into 1 data frame
rider_gas= pd.merge(rider,gas)
rider_weather=pd.merge(rider_gas,weather)
data=pd.merge(rider_weather,smartrip)


# Make it easier to work with columns by changing some column names.

data = data.rename(columns={'Mean TemperatureF':'Temp','PrecipitationIn':'Rain'})

# fill missing values in rain with 0
data.Rain.fillna(value=0) 
data.Rain.replace('T',0,inplace=True)
# add a new column that turns weekdays and weekends into dummy variables. 
data['Weekday']=data.Day.map({'Saturday':0,'Sunday':0,'Monday':1,'Tuesday':1,'Wednesday':1,'Thursday':1,'Friday':1})

#onvert Rain to a float
data.Rain.convert_objects(convert_numeric=True)

#scatter plot using several of the features
data.head()
sns.pairplot(data, x_vars=['gasprice','Temp','CPI','Weekday','Smartrip'], y_vars='Ridership')

# Adding a line
sns.pairplot(data, x_vars=['gasprice','Temp','CPI','Weekday','Smartrip'], y_vars='Ridership',size=6, aspect=0.7,kind='reg')

#Resetting the Index to date
data = data.set_index('Date')
# correlation matrix
data.corr()
# scatter plot matrix
sns.pairplot(data, kind='reg')

sns.heatmap(data.corr())

#measuring the realationship between gas prices and ridership
data.plot(kind='scatter',x='gasprice',y='Ridership')

#Measuring the relationship between smartcard benefits and ridership
data.plot(kind='scatter',x='Smartrip',y='Ridership')

#time series for ridership
total_cols=['Date','Ridership']
rider_total=data[total_cols]
rider_total.head()
rider_total.plot(label='Ridership')
rider_total=rider_total.tail(365)

# display correlation matrix in Seaborn using a heatmap
sns.heatmap(data.corr())

#Describing the model
lm = smf.ols(formula='Ridership ~ gasprice + Temp + Weekday + CPI + Smartrip', data=data).fit()

#How much does our X data tell us about the Y value
lm.rsquared

#Finding the Y intercept and coefficients. 
feature_cols=['gasprice','Temp','Weekday','CPI','Smartrip']
X=data[feature_cols]
feature_colsy=['Ridership']
y = data[feature_colsy]

#split the data into test and train data to see the accuracy of my model

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print X_train.shape
print X_test.shape

print y_train.shape
print y_test.shape

linreg = LinearRegression()
linreg.fit(X_train, y_train)

print linreg.intercept_
print linreg.coef_

#Creating an array of the Y Prediction

y_pred = linreg.predict(X_test)
y_pred = y_pred.round()


#Mean Absolute Error (MAE)
print metrics.mean_absolute_error(y_test,y_pred)
#Mean Sqauared Error (MSE)
print metrics.mean_squared_error(y_test,y_pred) 
# Root Mean Squared Error (RMSE)
print np.sqrt(metrics.mean_absolute_error(y_test,y_pred))

metrics.r2_score(y_test, y_pred)

ridertotal = pd.read_csv('ridertotal2.csv')

data_tail=data.tail(365)