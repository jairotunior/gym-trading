import pandas as pd
import numpy as np
import settings
from datetime import datetime


df = pd.read_csv(settings.DATA_DIRECTORY + "/ES/1Min/last.csv", sep=";")

df['dateTime'] = pd.to_datetime(df['dateTime'], format="%Y-%m-%d %H:%M:%S")
df['min'] = df['dateTime']
df['max'] = df['dateTime']
df = df.set_index('dateTime')

df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1, inplace=True)

conversion = {'min': 'min', 'max': 'max'}

df = df.resample('D').agg(conversion)
df = df.dropna(axis=0)

df['avaliable'] = df['max'] - df['min']
df['avaliable'] = [i.seconds // 60 for i in df['avaliable']]
df['avaliable'] = np.clip(df['avaliable'] - 120, 0, np.inf)

print(df[['avaliable']].head())
print(df.dtypes)

df.to_csv(settings.DATA_DIRECTORY + "/ES/dates.csv", sep=";")


"""
ranges = pd.date_range('01-01-2010', '31-12-2010')
data = pd.read_csv('G:/ES.csv', sep=';', names=['dateTime', 'price', 'volume'], header=None)
#data['dateTime'] = data.to_datetime(df['dateTime'], format="%Y%m%d %H%H%S %f")

data = data.set_index('dateTime')

print(data.iloc['01-01-2010':'31-12-2010', :].head())
"""