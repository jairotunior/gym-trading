import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import settings
import time
import random

from typing import Dict


# Formato del archivo
# dateTime, open, high, low, close, volume, cons, first_tick, last_tick
def convert_data_ohlc(path, symbol, type, frame, output, sep=";"):
    df = pd.read_csv(path, sep=sep)

    df['date'] = pd.to_datetime(df['dateTime'], utc=True, infer_datetime_format=True)
    df['date'] = df.date.values.astype(np.int64) // 10 ** 0
    df = df.set_index('date')

    df['tick_count'] = df.tick_count.values.astype(np.int)

    df['last_tick'] = df['tick_count'].cumsum() - 1
    df['first_tick'] = df['last_tick'].shift(1)
    df.fillna(-1, inplace=True)
    df['first_tick'] = df['first_tick'] + 1
    df['first_tick'] = df['first_tick'].astype('int64')

    print(df[['first_tick', 'last_tick']].head(10))

    lines = ["ohlc"
             + ",symbol=" + symbol
             + ",type=" + type
             + ",frame=" + str(frame)
             + " "
             + "open=" + str(row['open']) + ","
             + "high=" + str(row['high']) + ","
             + "low=" + str(row['low']) + ","
             + "close=" + str(row['close']) + ","
             + "volume=" + str(row['volume']) + ","
             + "cons=" + str(cons) + ","
             + "first_tick=" + str(row['first_tick']) + ","
             + "last_tick=" + str(row['last_tick'])
             + " {}".format(index) for cons, (index, row) in enumerate(df.iterrows())
    ]

    file = open(output, 'w', newline='')

    file.write("# DDL\n")
    file.write('CREATE DATABASE import\n')
    file.write('# DML\n')
    file.write('# CONTEXT-DATABASE: import\n')

    for item in lines:
        file.write("{}\n".format(item))

    file.close()


def convert_tick_data(path: str, symbol: str, output: str, columns: Dict[None, list]=None, format="%m/%d/%Y %H:%M:%S.%f"):

    file_in = open(path)

    file_out = open(output, 'w', newline='')

    file_out.write("# DDL\n")
    file_out.write('CREATE DATABASE import\n')
    file_out.write('# DML\n')
    file_out.write('# CONTEXT-DATABASE: import\n')

    last_date = None
    counter = 0

    for i, row in enumerate(file_in):
        line = row.replace("\n", "").split(";")

        if i == 0:
            date_index = columns.index('date')
            price_index = columns.index('price')
            volume_index = columns.index('volume')

        date = datetime.strptime(line[date_index][:-1], format)

        if last_date == date:
            print("IGUALES")

        if last_date == date:
            #counter = counter + 1
            #line[date_index] = np.int64((date + timedelta(0, 0, 0, 1 * counter)).timestamp() * 1000000000)
            line[date_index] = np.int64(date.timestamp() * 1000000000)
        else:
            #counter = 0
            line[date_index] = np.int64(date.timestamp() * 1000000000)

        format_line = "tick,symbol={} price={},volume={}".format(symbol, line[price_index], line[volume_index]) + ",cons={} ".format(str(i)) + str(line[date_index])
        #print(format_line)
        file_out.write("{}\n".format(format_line))

        last_date = date

        if i % 1000000 == 0:
            print("{}: {}".format(i, date))

    file_out.close()
    file_in.close()


def convert_tick_with_dates(path: str, output: str, columns: Dict[None, list]=None, format="%m/%d/%Y %H:%M:%S.%f"):

    file_in = open(path)

    file_out = open(output, 'w', newline='')

    last_date = None
    counter = 0

    for i, row in enumerate(file_in):
        line = row.replace("\n", "").split(";")

        if i == 0:
            date_index = columns.index('date')
            price_index = columns.index('price')
            volume_index = columns.index('volume')

        date = datetime.strptime(line[date_index][:-1], format)

        if last_date == date:
            counter = counter + 1
            line[date_index] = date + timedelta(0, 0, 1 * counter, 0)
        else:
            counter = 1
            line[date_index] = date + timedelta(0, 0, 1 * counter, 0)

        format_line = "{};{};{}".format(*line)
        #print(format_line)
        file_out.write("{}\n".format(format_line))

        last_date = date

        if i % 1000000 == 0:
            print("{}: {}".format(i, date))

    file_out.close()
    file_in.close()


def get_data_in_file(path, columns=Dict[None, list], format="%Y-%m-%d %H:%M:%S.%f"):

    file = open(path)

    if columns is not None:
        date_index = columns.index('date')
        price_index = columns.index('price')
        volume_index = columns.index('volume')

    for i, row in enumerate(file):
        line = row.replace("\n", "").split(";")

        date = datetime.strptime(line[date_index], format)

        if date >= datetime(2011, 1, 6, 10, 31, 0) and date <= datetime(2011, 1, 6, 10, 31, 15):
            print("{}: Price: {} - Volume: {} - Cons: {}".format(date, line[price_index], line[volume_index], i))

        if i % 1000000 == 0:
            print("{}: {}".format(i, date))


def convert_data_ohlc_esp(path, symbol, type, frame, output, sep=";"):
    df = pd.read_csv(path, sep=sep)

    df['dateTime'] = df['Date'].map(str) + ' ' + df['Timestamp'].map(str)
    df['date'] = pd.to_datetime(df['dateTime'])
    df['date'] = df.date.values.astype(np.int64) // 10 ** 0
    df = df.set_index('date')

    lines = ["ohlc"
             + ",symbol=" + symbol
             + ",type=" + type
             + ",frame=" + str(frame)
             + " "
             + "open=" + str(row['Open']) + ","
             + "high=" + str(row['High']) + ","
             + "low=" + str(row['Low']) + ","
             + "close=" + str(row['Close']) + ","
             + "volume=" + str(row['Volume']) + ","
             + "cons=" + str(cons)
             + " {}".format(index) for cons, (index, row) in enumerate(df.iterrows())
    ]

    file = open(output, 'w', newline='')

    file.write("# DDL\n")
    file.write('CREATE DATABASE import\n')
    file.write('# DML\n')
    file.write('# CONTEXT-DATABASE: import\n')

    for item in lines:
        file.write("{}\n".format(item))

    file.close()



if __name__ == "__main__":
    # OHLC Example
    #path = "/media/user/Datos/DATA TRADING/Raw/ES/ES LLC MD.csv"
    #symbol = 'ES'
    #type = "llc_v3"
    #frame = 3
    #output = '/media/user/Datos/DATA TRADING/InfluxDB/ES/{}-{}-{} MD.txt'.format(symbol, type, frame)
    #convert_data_ohlc(path, symbol, type, frame, output, sep=",")


    path = "F:/DATA TRADING/Raw/EURUSD/EURUSD Min1440.csv"
    symbol = 'EURUSD'
    type = "Min"
    frame = 1440
    output = 'F:/DATA TRADING/InfluxDB/EURUSD/{} {}{}.csv'.format(symbol, type, frame)
    convert_data_ohlc_esp(path, symbol, type, frame, output, sep=",")


    # Tick by Tick Example
    #csv_filepath = "/media/user/Datos/DATA TRADING/Raw/ES/ES Ticks MD.csv"
    #symbol = 'ES'
    #output = '/media/user/Datos/DATA TRADING/InfluxDB/{}/{} Ticks MD.csv'.format(symbol, symbol)
    #get_data_in_file(csv_filepath, columns=['price', 'volume', 'date'], format="%Y-%m-%d %H:%M:%S.%f")
    #convert_tick_data(path=csv_filepath, output=output, symbol=symbol, columns=['price', 'volume', 'date'], format="%Y-%m-%d %H:%M:%S.%f")

    """
    csv_filepath = "F:/DATA TRADING/Raw/EURUSD/EURUSD2018-2019/EURUSD_TICK2019.csv"
    symbol = 'EURUSD'
    output = "F:/DATA TRADING/InfluxDB/{}/{} Ticks 2019.csv".format(symbol, symbol)
    convert_tick_data(path=csv_filepath, output=output, symbol=symbol, columns=['date', 'price', 'volume'], format='%Y%m%d %H:%M:%S.%f')
    """

    # Convert Tick by Tick to Tick by Tick Modified Dates
    #path = "/media/user/Datos/DATA TRADING/Raw/ES/ES Ticks.csv"
    #output = '/media/user/Datos/DATA TRADING/Raw/ES/ES Ticks MD.csv'
    #convert_tick_with_dates(path=path, output=output, columns=['price', 'volume', 'date'], format="%Y%m%d %H:%M:%S.%f")

    """
    lista = [
        {'type': 'Min', 'frame': 15},
        #{'type': 'Min', 'frame': 60},
        #{'type': 'D', 'frame': 1},
        #{'type': 'W', 'frame': 1}
    ]

    symbol = 'ES'

    for i in lista:
        path = "/media/user/Datos/DATA TRADING/Raw/ES/ES {}{}.csv".format(i['type'], i['frame'])
        output = '/media/user/Datos/DATA TRADING/InfluxDB/ES/ES {}{}.txt'.format(i['type'], i['frame'])
        convert_data_ohlc(path, symbol, i['type'], i['frame'], output, sep=",")
    """

    """
    df = pd.read_csv("F:/DATA TRADING/Raw/EURUSD/EURUSD Ticks.csv", sep=";", names=['dateTime', 'price', 'volume'])

    print(df.head(5))
    print(df.tail(5))

    df['time'] = pd.to_datetime(df['dateTime'], format='%Y%m%d %H:%M:%S.%f')
    df = df.set_index('time')

    print(df.head(5))
    print(df.tail(5))
    """
