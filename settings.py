import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIRECTORY = os.path.join(ROOT_DIR, 'gym_trading/data')

DEFAULT_FORMAT = "%Y-%m-%d %H:%M:%S"  # "%Y-%m-%d %H:%M:%S.%f"
DEFAULT_TIME_ZONE = 'America/Bogota' #'US/Eastern'

ENABLE_TICK_BY_TICK_STEP = False


CAMBIOS_HORARIOS = {
    '1988':	{'horario_verano': '1988/04/03 09:00:00', 'horario_invierno': '1988/10/30 07:00:00'},
    '1989':	{'horario_verano': '1989/04/02 09:00:00', 'horario_invierno': '1989/10/29 07:00:00'},
    '1990':	{'horario_verano': '1990/04/01 09:00:00', 'horario_invierno': '1990/10/28 07:00:00'},
    '1991':	{'horario_verano': '1991/04/07 09:00:00', 'horario_invierno': '1991/10/27 07:00:00'},
    '1992':	{'horario_verano': '1992/04/05 09:00:00', 'horario_invierno': '1992/10/25 07:00:00'},
    '1993':	{'horario_verano': '1993/04/04 09:00:00', 'horario_invierno': '1994/10/31 07:00:00'},
    '1994':	{'horario_verano': '1994/04/03 09:00:00', 'horario_invierno': '1994/10/30 07:00:00'},
    '1995':	{'horario_verano': '1995/04/02 09:00:00', 'horario_invierno': '1995/10/29 07:00:00'},
    '1996':	{'horario_verano': '1996/04/07 09:00:00', 'horario_invierno': '1996/10/27 07:00:00'},
    '1997':	{'horario_verano': '1997/04/06 09:00:00', 'horario_invierno': '1997/10/27 07:00:00'},
    '1998':	{'horario_verano': '1998/04/05 09:00:00', 'horario_invierno': '1998/10/25 07:00:00'},
    '1999':	{'horario_verano': '1999/04/04 09:00:00', 'horario_invierno': '1999/10/31 07:00:00'},
    '2000':	{'horario_verano': '2000/04/02 09:00:00', 'horario_invierno': '2000/10/29 07:00:00'},
    '2001':	{'horario_verano': '2001/04/01 09:00:00', 'horario_invierno': '2001/10/28 07:00:00'},
    '2002':	{'horario_verano': '2002/04/07 09:00:00', 'horario_invierno': '2002/10/27 07:00:00'},
    '2003':	{'horario_verano': '2003/04/06 09:00:00', 'horario_invierno': '2003/10/26 07:00:00'},
    '2004':	{'horario_verano': '2004/04/04 09:00:00', 'horario_invierno': '2004/10/31 07:00:00'},
    '2005':	{'horario_verano': '2005/04/03 09:00:00', 'horario_invierno': '2005/10/30 07:00:00'},
    '2006':	{'horario_verano': '2006/04/02 09:00:00', 'horario_invierno': '2006/10/29 07:00:00'},
    '2007':	{'horario_verano': '2007/03/11 08:00:00', 'horario_invierno': '2007/11/04 07:00:00'},
    '2008':	{'horario_verano': '2008/03/09 08:00:00', 'horario_invierno': '2008/11/02 07:00:00'},
    '2009':	{'horario_verano': '2009/03/08 08:00:00', 'horario_invierno': '2009/11/01 07:00:00'},
    '2010':	{'horario_verano': '2010/03/14 08:00:00', 'horario_invierno': '2010/11/07 07:00:00'},
    '2011':	{'horario_verano': '2011/03/13 08:00:00', 'horario_invierno': '2011/11/06 07:00:00'},
    '2012':	{'horario_verano': '2012/03/11 08:00:00', 'horario_invierno': '2012/11/04 07:00:00'},
    '2013':	{'horario_verano': '2013/03/10 08:00:00', 'horario_invierno': '2013/11/03 07:00:00'},
    '2014':	{'horario_verano': '2014/03/09 08:00:00', 'horario_invierno': '2014/11/02 07:00:00'},
    '2015':	{'horario_verano': '2015/03/08 08:00:00', 'horario_invierno': '2015/11/01 07:00:00'},
    '2016':	{'horario_verano': '2016/03/13 08:00:00', 'horario_invierno': '2016/11/06 07:00:00'},
    '2017':	{'horario_verano': '2017/03/12 08:00:00', 'horario_invierno': '2017/11/05 07:00:00'},
    '2018':	{'horario_verano': '2018/03/11 08:00:00', 'horario_invierno': '2018/11/04 07:00:00'},
    '2019':	{'horario_verano': '2019/03/10 08:00:00', 'horario_invierno': '2019/11/03 07:00:00'},
}


HOLIDAYS = {
    '01/01/2019': "New Year's Day",
    '21/01/2019': "Martin Luther King, Jr. Day",
    '18/02/2019': "Washington's Birthday (Presidents' Day)",
    '19/04/2019': "Good Fridays",
    '27/05/2019': "Memorial Day",
    '04/07/2019': "Independence Day",
    '02/09/2019': "Labor Day",
    '28/11/2019': "Thanksgiving",
    '25/12/2019': "Christmas",
    '01/01/2018': "New Year's Day",
    '16/01/2018': "Martin Luther King, Jr. Day",
    '20/02/2018': "Washington's Birthday (Presidents' Day)",
    '14/04/2018': "Good Fridays",
    '29/05/2018': "Memorial Day",
    '04/07/2018': "Independence Day",
    '04/09/2018': "Labor Day",
    '23/11/2018': "Thanksgiving",
    '25/12/2018': "Christmas",
    '02/01/2017': "New Year's Day",
    '16/01/2017': "Martin Luther King, Jr. Day",
    '20/02/2017': "Washington's Birthday (Presidents' Day)",
    '14/04/2017': "Good Fridays",
    '29/05/2017': "Memorial Day",
    '04/07/2017': "Independence Day",
    '04/09/2017': "Labor Day",
    '23/11/2017': "Thanksgiving",
    '25/12/2017': "Christmas",
    '01/01/2016': "New Year's Day",
    '18/01/2016': "Martin Luther King, Jr. Day",
    '15/02/2016': "Washington's Birthday (Presidents' Day)",
    '25/03/2016': "Good Fridays",
    '30/05/2016': "Memorial Day",
    '04/07/2016': "Independence Day",
    '05/09/2016': "Labor Day",
    '24/11/2016': "Thanksgiving",
    '26/12/2016': "Christmas",
    '01/01/2015': "New Year's Day",
    '19/01/2015': "Martin Luther King, Jr. Day",
    '16/02/2015': "Washington's Birthday (Presidents' Day)",
    '03/04/2015': "Good Fridays",
    '25/05/2015': "Memorial Day",
    '03/07/2015': "Independence Day",
    '07/09/2015': "Labor Day",
    '26/11/2015': "Thanksgiving",
    '25/12/2015': "Christmas",
    '01/01/2014': "New Year's Day",
    '20/01/2014': "Martin Luther King, Jr. Day",
    '17/02/2014': "Washington's Birthday (Presidents' Day)",
    '18/04/2014': "Good Fridays",
    '26/05/2014': "Memorial Day",
    '04/07/2014': "Independence Day",
    '01/09/2014': "Labor Day",
    '27/11/2014': "Thanksgiving",
    '25/12/2014': "Christmas",
    '01/01/2013': "New Year's Day",
    '21/01/2013': "Martin Luther King, Jr. Day",
    '18/02/2013': "Washington's Birthday (Presidents' Day)",
    '29/03/2013': "Good Fridays",
    '27/05/2013': "Memorial Day",
    '04/07/2013': "Independence Day",
    '02/09/2013': "Labor Day",
    '28/11/2013': "Thanksgiving",
    '25/12/2013': "Christmas",
    '02/01/2012': "New Year's Day",
    '16/01/2012': "Martin Luther King, Jr. Day",
    '20/02/2012': "Washington's Birthday (Presidents' Day)",
    '06/04/2012': "Good Fridays",
    '28/05/2012': "Memorial Day",
    '04/07/2012': "Independence Day",
    '03/09/2012': "Labor Day",
    '22/11/2012': "Thanksgiving",
    '25/12/2012': "Christmas",
    '01/01/2011': "New Year's Day",
    '17/01/2011': "Martin Luther King, Jr. Day",
    '21/02/2011': "Washington's Birthday (Presidents' Day)",
    '22/04/2011': "Good Fridays",
    '30/05/2011': "Memorial Day",
    '04/07/2011': "Independence Day",
    '05/09/2011': "Labor Day",
    '24/11/2011': "Thanksgiving",
    '26/12/2011': "Christmas",
    '01/01/2010': "New Year's Day",
    '18/01/2010': "Martin Luther King, Jr. Day",
    '15/02/2010': "Washington's Birthday (Presidents' Day)",
    '02/04/2010': "Good Fridays",
    '31/05/2010': "Memorial Day",
    '05/04/2010': "Independence Day",
    '06/09/2010': "Labor Day",
    '25/11/2010': "Thanksgiving",
    '24/12/2010': "Christmas",
    '01/01/2009': "New Year's Day",
    '19/01/2009': "Martin Luther King, Jr. Day",
    '16/02/2009': "Washington's Birthday (Presidents' Day)",
    '10/04/2009': "Good Fridays",
    '25/05/2009': "Memorial Day",
    '03/07/2009': "Independence Day",
    '07/09/2009': "Labor Day",
    '26/11/2009': "Thanksgiving",
    '25/12/2009': "Christmas",
    '01/01/2008': "New Year's Day",
    '21/01/2008': "Martin Luther King, Jr. Day",
    '18/02/2008': "Washington's Birthday (Presidents' Day)",
    '21/03/2008': "Good Fridays",
    '26/05/2008': "Memorial Day",
    '04/07/2008': "Independence Day",
    '01/09/2008': "Labor Day",
    '27/11/2008': "Thanksgiving",
    '25/12/2008': "Christmas",
    '01/01/2007': "New Year's Day",
    '15/01/2007': "Martin Luther King, Jr. Day",
    '19/02/2007': "Washington's Birthday (Presidents' Day)",
    '06/04/2007': "Good Fridays",
    '28/05/2007': "Memorial Day",
    '04/07/2007': "Independence Day",
    '03/09/2007': "Labor Day",
    '22/11/2007': "Thanksgiving",
    '25/12/2007': "Christmas",

    # Days without data
    "2010-01-04": "",
    "2010-03-08": "",
    "2010-03-29": "",
    "2010-07-05": "",
    "2011-02-02": "",
    "2012-10-29": "",
    "2012-10-30": "",
    "2013-03-07": "",
    "2013-11-11": "",
    "2013-12-30": "",
    "2014-05-13": "",
    "2014-06-02": "",
    "2014-06-04": "",
    "2014-08-18": "",
    "2016-07-20": "",
    "2017-06-05": "",
    "2017-09-19": "",
    "2018-01-01": "",
    "2018-01-15": "",
    "2018-03-30": "",
    "2018-08-07": "",
    "2018-09-03": "",
    "2018-12-05": ""
}


"""
*********** Alias	Description ***************
    B	business day frequency
    C	custom business day frequency
    D	calendar day frequency
    W	weekly frequency
    M	month end frequency
    SM	semi-month end frequency (15th and end of month)
    BM	business month end frequency
    CBM	custom business month end frequency
    MS	month start frequency
    SMS	semi-month start frequency (1st and 15th)
    BMS	business month start frequency
    CBMS	custom business month start frequency
    Q	quarter end frequency
    BQ	business quarter end frequency
    QS	quarter start frequency
    BQS	business quarter start frequency
    A, Y	year end frequency
    BA, BY	business year end frequency
    AS, YS	year start frequency
    BAS, BYS	business year start frequency
    BH	business hour frequency
    H	hourly frequency
    T, min	minutely frequency
    S	secondly frequency
    L, ms	milliseconds
    U, us	microseconds
    N	nanoseconds
"""
DEFAULT_FREQUENCY_LIST = {
    'D': 'daily',
    'W': 'weekly',
    'M': 'monthly',
    'H': 'hourly',
    'T': 'Min',
    'Min': 'Min',
    "tick": 'tick'
}

FREQUENCY_LIST = {
    **DEFAULT_FREQUENCY_LIST,
}

SYMBOL_LIST = {
    'ES': {'tickSize': 0.25, 'tickValue': 12.5, 'commisions' : 2.00},
    'EURUSD': {'tickSize': 0.00001, 'tickValue': 1, 'commisions' : 2.00}
}



SYMBOL = 'ES'

START_DATE = "2018-10-09 00:00:00"
END_DATE = "2018-10-31 23:59:00"

# Account Money
ACCOUNT_MONEY = 100000

CONTRACTS = 10


FREQ_SHORT_TERM = {
    'type': 'Min',
    'frame': 1,
    'backperiods': 20,
    'timesteps': 0,
}

FREQ_LONG_TERM = [
    {
        'type': 'T',
        'frame': 60,
        'backperiods': 20,
        'timesteps': 0,
    }
]


INFLUXDB_CONNECTION = {
    'host': "127.0.0.1",
    'username': 'admin',
    'password': 'jj881203',
    'port': 8086,
    'database': 'import',
    'time_zone': DEFAULT_TIME_ZONE
}

INFLUXDB_DEFAULT_SERIES_PER_INSTRUMENT = [
    {'frame': '1', 'type': 'Min'},
    {'frame': '1', 'type': 'D'},
    {'frame': '1', 'type': 'tick'},
    {'frame': '3', 'type': 'llc_v3'},
    {'frame': '60', 'type': 'Min'},
    {'frame': '1440', 'type': 'Min'},
    {'frame': '240', 'type': 'Min'},
    {'frame': '21', 'type': 'Min'},
    {'frame': '30', 'type': 'Min'},
    {'frame': '15', 'type': 'Min'},
    {'frame': '1', 'type': 'W'}
]

"""
json = [
    {
        "measurement": "es",
        "tags": {
            "symbol": "ES",
            "type": "Min",
            "frame": 5
        },
        "time": "2010-11-10T23:00:00Z",
        "fields": {
            "open": 2474.25,
            "high": 2487.00,
            "low": 2470.00,
            "close": 2780.25,
            "volume": 1500
        }
    }
]
"""