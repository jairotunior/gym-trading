import pandas as pd
import numpy as np
from datetime import datetime
from influxdb import InfluxDBClient
from collections import deque


import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.offline as py_offline
import plotly


def plot_candlestick2(df):
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
        visible=True,
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

    f = go.FigureWidget(data=fig.data, layout=fig.layout)

    def zoom(layout, xrange):
        in_view = df.loc[fig.layout.xaxis.range[0]:fig.layout.xaxis.range[1]]
        fig.layout.yaxis.range = [in_view.high.min() - 10, in_view.high.max() + 10]

    fig.layout.on_change(zoom, 'xaxis.range')


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
        visible=True,
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



def llc_v3(path, ticks, trend_factor, counter_trend_factor, tick_size, output=None, format="%Y%m%d %H:%M:%S.%f"):

    file = open(path)

    # Candle Parameters
    candle_color = deque(maxlen=2)

    upper_limit = 0
    lower_limit = 0
    last_tick = None


    new_ohlc = pd.DataFrame([], columns=['dateTime', 'open', 'high', 'low', 'close', 'volume'])


    list_ticks = []


    for i, row in enumerate(file):
        line = row.replace("\n", "").split(";")

        price = float(line[0])
        volume = int(line[1])
        date = datetime.strptime(line[2], format)

        current_tick = {'date': date, 'price': price, 'volume': volume}

        # Agrega el tick a la lista de Ticks para crear la vela
        list_ticks.append(current_tick)


        if i == 0:
            # Define Tick de la primera vela
            upper_limit = price + (ticks * tick_size * trend_factor)
            lower_limit = price - (ticks * tick_size * trend_factor)

            #print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

        if last_tick is not None:
            if current_tick['date'].day != last_tick['date'].day:
                # Cierra la vela del dia anterior
                # La ultima vela del dia se ajusta al open de la vela del siguiente dia
                new_ohlc['close'].iloc[-1] = price

                if price > new_ohlc.iloc[-1, :]['high']:
                    new_ohlc['high'].iloc[-1] = price
                elif price < new_ohlc.iloc[-1, :]['low']:
                    new_ohlc['low'].iloc[-1] = price

                df_ticks = pd.DataFrame(list_ticks[:-1])

                condition = new_ohlc.shape[0] == 0

                ohlc = {
                    'dateTime': df_ticks.iloc[-1, :]['date'],
                    'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                    'high': df_ticks['price'].max(),
                    'low': df_ticks['price'].min(),
                    'close': df_ticks.iloc[-1, :]['price'],
                    'volume': df_ticks['volume'].sum(axis=0),
                    'tick_count': df_ticks['price'].count()
                }

                print("Candle: ", ohlc)

                new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

                # Define Tick de la primera vela
                if new_ohlc.iloc[-1, :]['close'] > new_ohlc.iloc[-1, :]['open']:
                    upper_limit = price + (ticks * tick_size * trend_factor)
                    lower_limit = price - (ticks * tick_size * counter_trend_factor)
                elif new_ohlc.iloc[-1, :]['close'] < new_ohlc.iloc[-1, :]['open']:
                    upper_limit = price + (ticks * tick_size * counter_trend_factor)
                    lower_limit = price - (ticks * tick_size * trend_factor)
                else:
                    upper_limit = price + (ticks * tick_size * trend_factor)
                    lower_limit = price - (ticks * tick_size * trend_factor)

                #print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)


        if price > upper_limit:

            df_ticks = pd.DataFrame(list_ticks)

            condition = new_ohlc.shape[0] == 0

            ohlc = {
                'dateTime': df_ticks.iloc[-1, :]['date'],
                'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                'high': df_ticks['price'].max(),
                'low': df_ticks['price'].min(),
                'close': price,
                'volume': df_ticks['volume'].sum(axis=0),
                'tick_count': df_ticks['price'].count()
            }

            print("Candle: ", ohlc)

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

            # Si es la primera vela agregarla otra vez
            if len(candle_color) < 2:
                for i in range(2 - len(candle_color)):
                    candle_color.append(candle)


            # Set new level price of the next candle
            if candle_color[0] == candle_color[1]:
                upper_limit = price + (ticks * tick_size * trend_factor)
                lower_limit = price - (ticks * tick_size * counter_trend_factor)
            else:
                upper_limit = price + (ticks * tick_size * trend_factor)
                lower_limit = price - (
                            abs(new_ohlc['close'].iloc[-1] - new_ohlc['open'].iloc[-1]) + (ticks * tick_size))


            #print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

            list_ticks = []

        elif price < lower_limit:

            df_ticks = pd.DataFrame(list_ticks)

            condition = new_ohlc.shape[0] == 0

            ohlc = {
                'dateTime': df_ticks.iloc[-1, :]['date'],
                'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                'high': df_ticks['price'].max(),
                'low': df_ticks['price'].min(),
                'close': price,
                'volume': df_ticks['volume'].sum(axis=0),
                'tick_count': df_ticks['price'].count()
            }

            print("Candle: ", ohlc)

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

            # new_ohlc = pd.concat([new_ohlc, ohlc], axis=0, join='outer', ignore_index=True)

            # Set new level price of the next candle
            if candle_color[0] == candle_color[1]:
                upper_limit = price + (ticks * tick_size * counter_trend_factor)
                lower_limit = price - (ticks * tick_size * trend_factor)
            else:
                upper_limit = price + (
                            abs(new_ohlc['open'].iloc[-1] - new_ohlc['close'].iloc[-1]) + (ticks * tick_size))
                lower_limit = price - (ticks * tick_size * trend_factor)


            #print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

            list_ticks = []

        last_tick = current_tick


    if output is not None:
        new_ohlc.to_csv(output)

    return new_ohlc


def range_bars(path, ticks, tick_size, output=None):
    file = open(path)

    # Candle Parameters
    candle_color = deque(maxlen=2)

    upper_limit = 0
    lower_limit = 0
    last_tick = None

    new_ohlc = pd.DataFrame([], columns=['dateTime', 'open', 'high', 'low', 'close', 'volume'])
    list_ticks = []

    for i, row in enumerate(file):
        line = row.replace("\n", "").split(";")

        price = float(line[0])
        volume = int(line[1])
        date = datetime.strptime(line[2][:-3], "%Y%m%d %H:%M:%S.%f")

        current_tick = {'date': date, 'price': price, 'volume': volume}

        # Agrega el tick a la lista de Ticks para crear la vela
        list_ticks.append(current_tick)


        if i == 0:
            # Define Tick de la primera vela
            upper_limit = price + ticks * tick_size
            lower_limit = price - ticks * tick_size

            print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

        if last_tick is not None:
            if current_tick['date'].day != last_tick['date'].day:
                # Cierra la vela del dia anterior
                # La ultima vela del dia se ajusta al open de la vela del siguiente dia
                #new_ohlc['close'].iloc[-1] = price

                if price > new_ohlc.iloc[-1, :]['high']:
                    new_ohlc['high'].iloc[-1] = price
                elif price < new_ohlc.iloc[-1, :]['low']:
                    new_ohlc['low'].iloc[-1] = price

                df_ticks = pd.DataFrame(list_ticks[:-1])

                condition = new_ohlc.shape[0] == 0

                ohlc = {
                    'dateTime': df_ticks.iloc[-1, :]['date'],
                    'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                    'high': df_ticks['price'].max(),
                    'low': df_ticks['price'].min(),
                    'close': df_ticks.iloc[-1, :]['price'],
                    'volume': df_ticks['volume'].sum(axis=0)
                }

                new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

                upper_limit = price + ticks * tick_size
                lower_limit = price - ticks * tick_size

                print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

        if price > upper_limit:

            df_ticks = pd.DataFrame(list_ticks[:-1])

            condition = new_ohlc.shape[0] == 0

            ohlc = {
                'dateTime': df_ticks.iloc[-1, :]['date'],
                'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                'high': df_ticks['price'].max(),
                'low': df_ticks['price'].min(),
                'close': df_ticks.iloc[-1, :]['price'],
                'volume': df_ticks['volume'].sum(axis=0)
            }

            print("Candle: ", ohlc)

            new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

            # Set new level price of the next candle
            upper_limit = price + ticks * tick_size
            lower_limit = price - ticks * tick_size

            print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

            list_ticks = []

        elif price < lower_limit:

            df_ticks = pd.DataFrame(list_ticks[:-1])

            condition = new_ohlc.shape[0] == 0

            ohlc = {
                'dateTime': df_ticks.iloc[-1, :]['date'],
                'open': df_ticks.iloc[0, :]['price'] if condition else new_ohlc.iloc[-1, :]['close'],
                'high': df_ticks['price'].max(),
                'low': df_ticks['price'].min(),
                'close': df_ticks.iloc[-1, :]['price'],
                'volume': df_ticks['volume'].sum(axis=0)
            }

            print("Candle: ", ohlc)

            new_ohlc = new_ohlc.append(ohlc, ignore_index=True)

            # Set new level price of the next candle
            upper_limit = price + ticks * tick_size
            lower_limit = price - ticks * tick_size

            print("Upper Limit: ", upper_limit, " - Lower Limit: ", lower_limit)

            list_ticks = []

        last_tick = current_tick

        if datetime(2010, 1, 1, 12, 0, 0).month < date.month:
            break

    if output is not None:
        new_ohlc.to_csv(output)

    return new_ohlc



if __name__ == "__main__":
    path = "/media/user/Datos/DATA TRADING/Raw/ES/ES Ticks MD.csv"
    output = "/media/user/Datos/DATA TRADING/Raw/ES/ES LLC MD.csv"
    ticks = 3
    trend_factor = 1
    counter_trend_factor = 4
    tick_size = 0.25

    candles = llc_v3(path=path, ticks=ticks, trend_factor=trend_factor, counter_trend_factor=counter_trend_factor, tick_size=tick_size, output=output, format="%Y-%m-%d %H:%M:%S.%f")
    #plot_candlestick(candles)

    #candles = range_bars(path=path, ticks=ticks, tick_size=tick_size)
    #plot_candlestick2(candles)