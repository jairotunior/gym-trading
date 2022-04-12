import pandas as pd


def resample_ohlc(df, type='T', frame=60):
    # Resample by Tick Frame
    conversion = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'tick_count': 'sum'}

    #df_ohlc = df.set_index('dateTime')
    df = df.resample("{}{}".format(frame, type), label='right').agg(conversion)
    #df = df.reset_index()
    df = df.dropna(axis=0)

    return df


path = "/media/user/Datos/DATA TRADING/Raw/ES/ES M1.csv"

df = pd.read_csv(path, sep=";")
df['dateTime'] = pd.to_datetime(df['Date and Time'], infer_datetime_format=True)

df['open'] = df['Open']
df['high'] = df['High']
df['low'] = df['Low']
df['close'] = df['Close']
df['volume'] = df['Volume']
df['tick_count'] = df['Tick Count']

df = df.set_index('dateTime')

tf = [
    {'type': 'Min', 'frame': 15},
    #{'type': 'Min', 'frame': 60},
    #{'type': 'D', 'frame': 1},
    #{'type': 'W', 'frame': 1}
]

for i in tf:
    resample = resample_ohlc(df[['open', 'high', 'low', 'close', 'volume', 'tick_count']], i['type'], i['frame'])
    resample.to_csv("{}{}.csv".format(i['type'], i['frame']))
