import pandas as pd
import numpy as np
import settings
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as py_offline
from collections import deque



def plot_candlestick(df):
    INCREASING_COLOR = '#17BECF'
    DECREASING_COLOR = '#7F7F7F'

    data = [dict(
        type='candlestick',
        open=df.open,
        high=df.high,
        low=df.low,
        close=df.close,
        x=df.index,
        yaxis='y2',
        name='GS',
        increasing=dict(line=dict(color=INCREASING_COLOR)),
        decreasing=dict(line=dict(color=DECREASING_COLOR)),
    )]

    layout = dict()

    fig = dict(data=data, layout=layout)

    fig['layout'] = dict()
    fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
    fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
    fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False)
    fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8])
    fig['layout']['legend'] = dict(orientation='h', y=0.9, x=0.3, yanchor='bottom')
    fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

    rangeselector = dict(
        visibe=True,
        x=0, y=0.9,
        bgcolor='rgba(150, 200, 250, 0.4)',
        font=dict(size=13),
        buttons=list([
            dict(count=1,
                 label='reset',
                 step='all'),
            dict(count=1,
                 label='1yr',
                 step='year',
                 stepmode='backward'),
            dict(count=3,
                 label='3 mo',
                 step='month',
                 stepmode='backward'),
            dict(count=1,
                 label='1 mo',
                 step='month',
                 stepmode='backward'),
            dict(step='all')
        ]))

    fig['layout']['xaxis']['rangeselector'] = rangeselector

    py_offline.plot(fig, filename='candlestick-test.html', validate=False)


if __name__ == "__main__":

    data = pd.read_csv("/media/user/Datos/DATA TRADING/Raw/ES/ES Ticks.csv", sep=";", names=['price', 'volume', 'dateTime'])
    data['dateTime'] = data['dateTime'][:-1]
    data['dateTime'] = pd.to_datetime(data['dateTime'], format="%Y%m%d %H:%M:%S.%f")
    #data['dateTime'] = pd.to_datetime(data['dateTime'], format="%Y%m%d %H%M%S %f")

    print(data.head())

    rows = data.shape[0]

    data = data.set_index('dateTime')
    conversion = {'min': 'min', 'max': 'max'}

    days = data.drop(['volume'], axis=1, inplace=False)
    days = days.resample('D').agg(conversion)
    days = days.dropna(axis=0)

    data = data.reset_index()
    days = days.reset_index()

    #print(days.head())
    candle_color = deque(maxlen=2)

    ticks = 3
    trend_factor = 1
    counter_trend_factor = 4
    tick_size = 0.01

    upper_limit = 0
    lower_limit = 0

    last_index = 0

    new_ohlc = pd.DataFrame([], columns=['dateTime', 'open', 'high', 'low', 'close', 'volume'])

    for j, date in days.iterrows():
        print(date['dateTime'][0])
        day_data = data[data['dateTime'].dt.date == date['dateTime'].dt.date[0]]
        day_data = day_data.reset_index()
        #print(day_data.head(20))
        last_index, i, new_candle = [0,0, True]
        for i, row in day_data.iterrows():
            if i == 0:
                if new_ohlc.shape[0] == 0:
                    upper_limit = row['price'] + (ticks * tick_size * trend_factor)
                    lower_limit = row['price'] - (ticks * tick_size * trend_factor)
                    last_index = i
                else:
                    # La ultima vela del dia se ajusta al open de la vela del siguiente dia
                    new_ohlc['close'].iloc[-1] = row['price']
                    # Todos los de abajo no funcionan para asignar valor
                    #new_ohlc.iloc[-1].at['close'] = row['price']
                    #new_ohlc.iloc[-1, :]['close'] = row['price']
                    #new_ohlc.loc[-1, 'close'] = row['price']
                    #new_ohlc.at[-1, 'close'] = row['price']
                    if row['price'] > new_ohlc.iloc[-1, :]['high']:
                        new_ohlc['high'].iloc[-1] = row['price']
                    elif row['price'] < new_ohlc.iloc[-1, :]['low']:
                        new_ohlc['low'].iloc[-1] = row['price']

                    if new_ohlc.iloc[-1, :]['close'] > new_ohlc.iloc[-1, :]['open']:
                        upper_limit = row['price'] + (ticks * tick_size * trend_factor)
                        lower_limit = row['price'] - (ticks * tick_size * counter_trend_factor)
                    else:
                        upper_limit = row['price'] + (ticks * tick_size * counter_trend_factor)
                        lower_limit = row['price'] - (ticks * tick_size * trend_factor)
                continue

            if row['price'] > upper_limit:
                candle_data = day_data.iloc[last_index:i+1, :]
                candle_data = candle_data.reset_index()

                print("Up: {} - Down: {} - i: {} - Last Index: {} - Value: {}".format(upper_limit, lower_limit, i, last_index, data.iloc[i, :]))
                #print(candle_data)

                condition = new_ohlc.shape[0] == 0 or new_candle == True

                ohlc = {
                    'dateTime': candle_data.iloc[-1, :]['dateTime'],
                    'open': candle_data.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                    'high': candle_data['price'].max(),
                    'low': candle_data['price'].min(),
                    #'close': candle_data.iloc[-1, :]['price'],
                    'close':row['price'],
                    'volume': candle_data['volume'].sum(axis=0)
                }

                print(ohlc)

                new_ohlc = new_ohlc.append(ohlc, ignore_index=True)
                #ohlc = pd.DataFrame(ohlc)

                # Green Candle
                candle = 1

                # Red Candle
                if ohlc['open'] > ohlc['close']:
                    candle = 2
                    # Doji
                elif ohlc['open'] == ohlc['close']:
                    candle = 0

                candle_color.append(candle)

                if len(candle_color) < 2:
                    for i in range(2 - len(candle_color)):
                        candle_color.append(candle)

                #new_ohlc = pd.concat([new_ohlc, ohlc], axis=0, join='outer', ignore_index=True)

                # Set new level price of the next candle
                if candle_color[0] == candle_color[1]:
                    upper_limit = row['price'] + (ticks * tick_size * trend_factor)
                    lower_limit = row['price'] - (ticks * tick_size * counter_trend_factor)
                else:
                    upper_limit = row['price'] + (ticks * tick_size * trend_factor)
                    lower_limit = row['price'] - (abs(new_ohlc['close'].iloc[-1] - new_ohlc['open'].iloc[-1]) + (ticks * tick_size))

                last_index = i
                new_candle = False
            elif row['price'] < lower_limit:
                candle_data = day_data.iloc[last_index:i+1, :]
                candle_data = candle_data.reset_index()

                #print("Up: {} - Down: {} - i: {} - Last Index: {} - Value: {}".format(upper_limit, lower_limit, i, last_index, data.iloc[i, :]))
                #print(candle_data)

                condition = new_ohlc.shape[0] == 0 or new_candle == True

                ohlc = {
                    'dateTime': candle_data.iloc[-1, :]['dateTime'],
                    'open': candle_data.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                    'high': candle_data['price'].max(),
                    'low': candle_data['price'].min(),
                    #'close': candle_data.iloc[-1, :]['price'],
                    'close': row['price'],
                    'volume': candle_data['volume'].sum(axis=0)
                }

                print(ohlc)

                new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

                # Green Candle
                candle = 1

                # Red Candle
                if ohlc['open'] > ohlc['close']:
                    candle = 2
                    # Doji
                elif ohlc['open'] == ohlc['close']:
                    candle = 0

                candle_color.append(candle)

                if len(candle_color) < 2:
                    for i in range(2 - len(candle_color)):
                        candle_color.append(candle)

                #new_ohlc = pd.concat([new_ohlc, ohlc], axis=0, join='outer', ignore_index=True)

                # Set new level price of the next candle
                if candle_color[0] == candle_color[1]:
                    upper_limit = row['price'] + (ticks * tick_size * counter_trend_factor)
                    lower_limit = row['price'] - (ticks * tick_size * trend_factor)
                else:
                    upper_limit = row['price'] + (abs(new_ohlc['open'].iloc[-1] - new_ohlc['close'].iloc[-1]) + (ticks * tick_size))
                    lower_limit = row['price'] - (ticks * tick_size * trend_factor)

                last_index = i
                new_candle = False

        new_candle = True

        """
        if last_index != i and row['dateTime'].date() != data.iloc[last_index, :]['dateTime'].date():
            candle_data = data.iloc[last_index:i, :]
            candle_data = candle_data.reset_index()

            print("out")

            print("Up: {} - Down: {} - i: {} - Value: {}".format(upper_limit, lower_limit, i, data.iloc[i, :]))
            print(candle_data)

            ohlc = {
                'dateTime': candle_data.iloc[-1, :]['dateTime'],
                'open': candle_data.iloc[0, :]['price'] if new_ohlc.shape[0] == 0 else new_ohlc.iloc[-1, :]['close'],
                'high': candle_data['price'].max(),
                'low': candle_data['price'].min(),
                'close': candle_data.iloc[-1, :]['price'],
                'volume': candle_data['volume'].sum(axis=0)
            }

            print(ohlc)

            new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

            upper_limit = row['price'] + (ticks * tick_size * counter_trend_factor)
            lower_limit = row['price'] - (ticks * tick_size * trend_factor)
        """


    print(new_ohlc)

    new_ohlc.to_csv("/media/user/Datos/DATA TRADING/Raw/ES/ES LLC.csv")
    #new_ohlc = new_ohlc.set_index('dateTime')

    plot_candlestick(new_ohlc)
