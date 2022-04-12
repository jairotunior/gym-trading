from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import settings
import os

from influxdb import InfluxDBClient
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np


def standard_ohlc(df, output):
    df['size_candle'] = df['close'] - df['open']
    df['gap_size'] = df['open'] - df['close'].shift(1)
    df['high_size'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['low_size'] = df[['open', 'close']].min(axis=1) - df['low']
    df = df.fillna(0)


    timeframes = [
        "3llc_v3",
    ]

    conversion = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    for tf in timeframes:
        df_tf = df
        #df_tf = df.resample("{}".format(tf), label='right').agg(conversion)
        #df_tf = df_tf.dropna(axis=0)

        df_tf['candle_size'] = df_tf['close'] - df_tf['open']
        df_tf['gap_size'] = df_tf['open'] - df_tf['close'].shift(1)
        df_tf['high_size'] = df_tf['high'] - df_tf[['open', 'close']].max(axis=1)
        df_tf['low_size'] = df_tf[['open', 'close']].min(axis=1) - df_tf['low']
        df_tf['volume_size'] = df_tf['volume']
        df_tf = df_tf.fillna(0)

        metrics = ['candle_size', 'gap_size', 'high_size', 'low_size', 'volume_size']

        for metric in metrics:
            data = df_tf[metric].values
            data = np.reshape(data, (*data.shape, 1))

            scaler = StandardScaler()
            data = scaler.fit_transform(data)

            joblib.dump(scaler, os.path.join(settings.ROOT_DIR, '{}/{}/scaler_{}_{}.pkl'.format(output, metric, metric, tf)))


if __name__ == "__main__":
    client = InfluxDBClient(
        'localhost',
        30000,
        'admin',
        'jj881203',
        'import'
    )

    result = client.query(
        "SELECT open, high, low, close, volume FROM import.autogen.ohlc WHERE symbol='ES' and type='llc_v3' and frame='3'")

    df = pd.DataFrame(list(result)[0])
    df['time'] = pd.to_datetime(df['time'], infer_datetime_format=True)
    df = df.set_index('time')

    df['candle_size'] = df['close'] - df['open']
    df['gap_size'] = df['open'] - df['close'].shift(1)
    df['high_size'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['low_size'] = df[['open', 'close']].min(axis=1) - df['low']
    df['volume_size'] = df['volume']
    df = df.fillna(0)

    metrics = ['candle_size', 'gap_size', 'high_size', 'low_size', 'volume_size']

    print(df[metrics].head(30))


    data = df['candle_size'].values
    data = np.reshape(data, (*data.shape, 1))

    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    df['candle_size'] = data


    df = df[metrics]

    print(df.describe())

    print("Candle_size **** Min: {} - Max: {}".format(df[df['candle_size'] < -1].count()['candle_size'], df[df['candle_size'] > 1].count()['candle_size']))

    df.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False)
    plt.show()

    """
    df.hist()
    plt.show()

    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
    plt.show()

    df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False)
    plt.show()
    """


    """
    data = df[metrics].values
    data = np.reshape(data, (*data.shape, 1))

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    """

    #output = "scalers"

    #standard_ohlc(df, output)