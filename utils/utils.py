import pytz
from datetime import datetime, timedelta
import numpy as np

import pandas as pd
import os

import settings
import time
import random


"""
 Convert dates to default format and timezone
"""
def convert_datetime_with_timezone(date, time_zone = settings.DEFAULT_TIME_ZONE, format_date=settings.DEFAULT_FORMAT):
    date = datetime.strptime(date, format_date)
    timezone = pytz.timezone(time_zone).localize(date)
    return timezone

"""
 Convert dates to default format and timezone
"""
def convert_datetime(date, format_date=settings.DEFAULT_FORMAT):
    date = datetime.strptime(date, format_date)
    return date


def get_dataframe(symbol, start, end, type='T', frame=1, sep=';', format_date=settings.DEFAULT_FORMAT, backperiods=20, serie=None):
    assert symbol in settings.SYMBOL_LIST, 'The symbol name is not registered in settings file.'
    assert isinstance(symbol, str)
    assert isinstance(start, str)
    assert isinstance(end, str)
    assert type in settings.FREQUENCY_LIST, "The frequence selected is unknown."
    assert isinstance(frame, int)
    assert isinstance(sep, str)
    assert backperiods > 0, "El parametro backperiods no puede ser igual o menor que cero"

    path = r"{}/{}/{}/{}.csv".format(settings.DATA_DIRECTORY, symbol, "{}{}".format(frame,settings.FREQUENCY_LIST[type]), 'last')

    if type == 'T' or type == 'Min' or type == 'H':
        path = r"{}/{}/{}/{}.csv".format(settings.DATA_DIRECTORY, symbol, "1Min", 'last')
    elif type == 'M' or type == 'W' or type == 'D':
        path = r"{}/{}/{}/{}.csv".format(settings.DATA_DIRECTORY, symbol, "1D", 'last')
    elif type == 'tick':
        pass

    #data = pd.read_csv(path, sep=sep, usecols=['open', 'high', 'low', 'close'], parse_dates=['dateTime'])
    data = pd.read_csv(path, sep=sep, usecols=['dateTime', 'open', 'high', 'low', 'close', 'volume'])

    data['dateTime'] = pd.to_datetime(data['dateTime'], format=format_date)


    if type == 'T' or type == 'Min':
        if frame > 1:
            data = resample_ohlc(data, type, frame)
    elif type == 'H':
        data = resample_ohlc(data, "Min", frame*60)
    elif type == 'D':
        if frame > 1:
            data = resample_ohlc(data, type, frame)
    elif type == 'M':
        data = resample_ohlc(data, type, frame)
    elif type == 'W':
        data = resample_ohlc(data, "")
    elif type == 'tick':
        pass

    data = data.dropna(axis=0)

    start_date = convert_datetime(start)
    #start_date = convert_datetime_with_timezone(start)

    #print("Offset: {}".format(start_date))

    start_date = get_base_bars(serie, start_date, type, frame, backperiods)

    #print("Start Date DF: {}".format(start_date))

    #print("Serie Before Filter")
    #print(data)

    mask = (data['dateTime'] >= start_date.strftime(settings.DEFAULT_FORMAT)) & (data['dateTime'] <= end)
    #return data[start: end]
    data = data.loc[mask]
    data = data.reset_index()

    #print(data)

    return data


"""
    The commisions are calculated by entry per side
"""
def get_commisions(symbol, contracts):
    return settings.SYMBOL_LIST[symbol]['commisions'] * contracts


def convert_backperiods_to_time(type, frame, backperiods):
    assert type in settings.FREQUENCY_LIST, "El tipo de temporalidad {}, no esta soportado".format(type)
    assert isinstance(backperiods, int)

    periods = backperiods*frame

    if type == 'T' or type == 'Min':
        return timedelta(minutes=periods)
    elif type == 'H':
        return timedelta(hours=periods)
    elif type == 'D':
        return timedelta(days=periods)
    elif type == 'M':
        return timedelta(days=periods * 30)
    elif type == 'W':
        return timedelta(weeks=periods)


"""
    The serie format is:
    ['dateTime', 'min', 'max', 'avaliable']
    avaliable params is in minutes
"""
def get_base_bars(serie, current_date, type, frame, backperiods):
    # Se obtiene el requerimiento de tiempo bruto

    time = 0
    discount_factor = timedelta(days=2)

    if type == 'T' or type == 'Min':
        time = timedelta(minutes=frame * backperiods)
    elif type == 'H':
        time = timedelta(hours=frame * backperiods)
    elif type == 'D':
        time = timedelta(days=frame * backperiods)
        discount_factor = timedelta(days=7)
    elif type == 'M':
        time = timedelta(days=frame * backperiods * 30)
        discount_factor = timedelta(days=30)
    elif type == 'W':
        time = timedelta(weeks=frame * backperiods)
        discount_factor = timedelta(weeks=1)

    # Se obtiene una lista de la disponibilidad de tiempo por dias
    current_date -= timedelta(days=1)
    start_date = current_date - time
    #print("Current Time: {} - Start Date: {}".format(current_date.date(), start_date.date()))

    mask = (serie['dateTime'] >= start_date.date()) & (serie['dateTime'] <= current_date.date())
    data = serie.loc[mask]

    data = data.dropna(axis=0)

    #print("Time Avaliable: {} - Time Need: {} - Start: {} - Current Date: {}".format(data['avaliable'].sum(), time.days * 24 * 60 + time.seconds / 60, start_date, current_date))

    # Itera hasta completar la cantidad de datos necesarias
    while data['avaliable'].sum() < (time.days * 24 * 60 + time.seconds / 60):
        #print("While: {}".format(data['avaliable'].sum()))
        start_date -= discount_factor
        #print("New Start Date: {} - Current Date: {}".format(start_date.date(), current_date.date()))
        mask = (serie['dateTime'] >= start_date.date()) & (serie['dateTime'] <= current_date.date())
        data = serie.loc[mask]
        data = serie.dropna(axis=0)

    #return data.loc[mask].iloc[0, 0]
    return data.iloc[0, 0]


def obtain_ohlc(path, name_price='ask', num_ticks = 34, position=''):

    df = pd.read_csv(
        path,
        usecols = ['dateTime', name_price],
        na_values = ['nan'],
        parse_dates = True
    )

    df["dateTime"] = pd.to_datetime(df["dateTime"], unit='ms')

    df = df.set_index('dateTime')

    ohlc = df[name_price].resample('1Min').ohlc()
    #ohlc = df.groupby(np.arange(len(df.index)) // num_ticks)

    #ohlc['sret'] = ohlc['close'].pct_change()
    ohlc['sret'] = np.log(ohlc['close']/ohlc['close'].shift())

    ohlc['sret_high'] = np.log(ohlc['high']/ohlc['close'].shift())
    ohlc['sret_low'] = np.log(ohlc['low']/ohlc['close'].shift())

    # Elimina los valores no numericos del dataset
    ohlc = ohlc.dropna()

    if position == 'high' or position == 'low':
        ohlc = ohlc[position]

    return ohlc


'''
    El formato del Dataframe debe ser el siguiente:
    dateTime - open - high - low - close

    No debe tener columna de indice de fecha
'''
def resample_ohlc(df, type='T', frame=60):
    # Resample by Tick Frame
    conversion = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}

    #df_ohlc = df.set_index('dateTime')
    df = df.resample("{}{}".format(frame, type), label='right').agg(conversion)
    #df = df.reset_index()
    df = df.dropna(axis=0)

    return df

def resample_ohlc_tick(df, tick_frame=34):
    # Resample by Tick Frame
    df['count'] = range(1, len(df) + 1)

    df['group'] = np.floor(df['count'] / tick_frame)

    ohlc = {'dateTime': df.groupby(['group'])['dateTime'].last(),
            'open': df.groupby(['group'])['ask'].first(),
            'high': df.groupby(['group'])['ask'].max(),
            'low': df.groupby(['group'])['ask'].min(),
            'close': df.groupby(['group'])['ask'].last(),
            # 'count': df.groupby(['group'])['ask'].count()
            }

    df_ohlc = pd.DataFrame(ohlc, columns=['dateTime', 'open', 'high', 'low', 'close'])

    df_ohlc = df_ohlc.set_index('dateTime')

    return df_ohlc


def save_tmp(df, symbol, type, frame):
    path = settings.DATA_DIRECTORY.join("/{}.h5".format(symbol))

    # Create data store
    data_store = pd.HDFStore(path)

    data_store["{}-{}".format(type, frame)] = df
    data_store.close()

def load_tmp(symbol, type, frame):
    path = settings.DATA_DIRECTORY.join("/{}.h5".format(symbol))

    # Access Data Store
    data_store = pd.HDFStore(path)

    if "{}-{}".format(type, frame) in data_store:
        df = data_store["{}-{}".format(type, frame)]
        data_store.close()

    return df


def strTimeProp(start, end, format, prop, output_format):
    start_time = time.mktime(time.strptime(start, format))
    end_time = time.mktime(time.strptime(end, format))

    new_date = start_time + prop * (end_time - start_time)

    return time.strftime(output_format, time.localtime(new_date))
    #return new_date


def random_date(start, end, format, output_format="%d/%m/%Y") -> datetime:
    date = strTimeProp(start, end, format, random.random(), output_format)
    holidays = settings.HOLIDAYS.keys()
    date_obj = datetime.strptime(date, output_format)

    while (date in holidays or date_obj.weekday() > 4):
        date = strTimeProp(start, end, format, random.random(), output_format)
        date_obj = datetime.strptime(date, output_format)

    return date
