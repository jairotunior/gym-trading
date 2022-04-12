import pandas as pd
import pytz
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil.parser import parse

import settings

class HistoricalConnection:

    def __init__(self, host='localhost', port = 8086, username='admin', password=None, database=None, time_zone = 'US/Eastern'):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.time_zone = time_zone

        self.client = InfluxDBClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database
        )

    def get_data(self, start_date=None, end_date=None, start_cons=None, end_cons=None, format = "%Y-%m-%d %H:%M:%S", symbol=None, type=None, frame=None, offset = 0):
        assert isinstance(symbol, str), "El argumento symbol debe ser de tipo str"
        assert isinstance(type, str), "El argumento type debe ser de tipo str"
        assert isinstance(frame, int), "El argumento frame debe ser de tipo int"

        if start_date is not None and end_date is not None:
            assert isinstance(start_date, str), "El parametro 'start_date' debe ser de tipo 'str'."
            assert isinstance(end_date, str), "El parametro 'end_date' debe ser de tipo 'str'."

            # Opcion 1: Convert str to datetime without timezone
            start_date = datetime.strptime(start_date, format)
            end_date = datetime.strptime(end_date, format)

            tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
            tz_utc = pytz.timezone('UTC')

            query_start_date = tz_utc.localize(start_date).astimezone(tz_utc)
            query_end_date = tz_utc.localize(end_date).astimezone(tz_utc)

            # Opcion 2: Convert str to datetime with timezone
            # start and end dates localized in time Zone
            # example: start = '2010-01-04 00:00:00' convert to '2010-01-04 00:00:00-05:00'
            # start_date = pytz.timezone(self.time_zone).localize(parse(start))
            # end_date = pytz.timezone(self.time_zone).localize(parse(end))

            result = self.select(measurement="ohlc",
                                 fields=['open', 'high', 'low', 'close', 'volume', 'cons'],
                                 dates=[query_start_date.strftime("%Y-%m-%dT%H:%M:%SZ"), query_end_date.strftime("%Y-%m-%dT%H:%M:%SZ")],
                                 symbol=symbol,
                                 type=type,
                                 frame=frame)
        elif start_cons is not None and end_cons is not None:
            assert isinstance(start_cons, int), "El parametro 'start_cons' debe ser de tipo 'int'."
            assert isinstance(end_cons, int), "El parametro 'end_cons' debe ser de tipo 'int'."

            result = self.select(measurement="ohlc",
                                 fields=['open', 'high', 'low', 'close', 'volume', 'cons'],
                                 cons=[start_cons + offset, end_cons],
                                 symbol=symbol,
                                 type=type,
                                 frame=frame)

        return result

    def select(self, measurement, fields, dates=None, cons=None, symbol=None, type=None, frame=None):
        assert isinstance(symbol, str), "El argumento symbol debe ser de tipo str"
        assert isinstance(type, str), "El argumento type debe ser de tipo str"
        assert isinstance(frame, int), "El argumento frame debe ser de tipo int"

        f = ",".join(fields)

        if dates is not None:
            query = "SELECT {} FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}' and time >= '{}' and time < '{}'".format(f, self.database, measurement, symbol, type, frame, *dates)
        elif cons is not None:
            query = "SELECT {} FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}' and cons >= {} and cons <= {}".format(f, self.database, measurement, symbol, type, frame, *cons)
        else:
            query = "SELECT {} FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}'".format(f, self.database, measurement, symbol, type, frame)

        try:
            result = self.client.query(query)
            df = pd.DataFrame(list(result)[0])
            df['dateTime'] = pd.to_datetime(df['time'])

            # Para modificar el time_zone a una serie que ya tiene una asignada
            # Primero se debe quitar y luego asignar la nueva
            df['dateTime'] = df['dateTime'].dt.tz_localize(None).dt.tz_localize('UTC')
            #df['dateTime'] = df['dateTime'].dt.tz_convert('UTC')
            df = df.set_index('dateTime')
        except:
            return None

        return df

    def get_first_cons(self, measurement, start=None, end=None, symbol=None, type=None, frame=None, format = "%Y-%m-%d %H:%M:%S", verbose=0):
        # Opcion 1: Convert str to datetime without timezone
        start_date = datetime.strptime(start, format)
        end_date = datetime.strptime(end, format)

        tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
        tz_utc = pytz.timezone('UTC')

        query_start_date = tz_utc.localize(start_date).astimezone(tz_utc)
        query_end_date = tz_utc.localize(end_date).astimezone(tz_utc)

        query = "SELECT first(cons), last(cons) FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}' and time >= '{}' and time < '{}'".format(
            self.database, measurement, symbol, type, frame, query_start_date.strftime("%Y-%m-%dT%H:%M:%SZ"), query_end_date.strftime("%Y-%m-%dT%H:%M:%SZ"))

        result = None
        try:
            result = list(self.client.query(query))[0]
        except IndexError:
            if verbose == 0:
                pass
            else:
                print("Result Empty in query: {}".format(query))

        return result

    def get_ticks(self, start, end, symbol, format="%Y-%m-%d %H:%M:%S.%f", verbose=0):
        start_date = datetime.strptime(start, format)
        end_date = datetime.strptime(end, format)

        query = "SELECT price, volume FROM {}.autogen.tick WHERE symbol='{}' and time > '{}' and time <= '{}'".format(
            self.database, symbol, start_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"), end_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))

        df = None
        try:
            result = list(self.client.query(query))[0]
            df = pd.DataFrame(result)
            df['dateTime'] = pd.to_datetime(df['time'], infer_datetime_format=True)
            #df['dateTime'] = df['dateTime'].dt.tz_localize(None).dt.tz_localize(self.time_zone)

            df['dateTime'] = df['dateTime'].dt.tz_localize(None).dt.tz_localize('UTC')
            #df['dateTime'] = df['dateTime'].dt.tz_convert('UTC')
            df = df.set_index('dateTime')
        except IndexError:
            if verbose == 0:
                pass
            else:
                print("Result Empty in query: {}".format(query))

        return df

    def query(self, query):
        return self.client.query(query)

if __name__ == "__main__":
    conn = HistoricalConnection(
        host='45.55.39.163',
        port=8086,
        username='admin',
        password='4ee89216417f73eee27565cd6ec162061aa83dcd427f96e4',
        database='financial_data'
    )

    START_DATE = "2010-01-30 00:00:00"
    END_DATE = "2010-03-16 23:59:00"

    data = conn.get_data(start_date=START_DATE, end_date=END_DATE, symbol='ES', type='Min', frame=1)
    print(data['close'].values)

    print(list(conn.get_first_cons(measurement='ohlc', start=START_DATE, end=END_DATE, symbol='ES', type='Min', frame=1))[0])