import gym
import os
from pathlib import Path
import settings
import joblib
from gym_trading.envs import TradingEnv
from utils.influxdb import HistoricalConnection
from gym_trading.wrappers import PreprocessObservation, RewardRiskRatios, RewardMarketRatios
from utils import random_date
from datetime import datetime, timedelta

import matplotlib.pyplot as plt


import logging


def create_environment_function_2(symbol, info_st, info_lt, connection : HistoricalConnection):

    info = []
    info.append(info_st)

    if info_lt is not None:
        for lt in info_lt:
            info.append(lt)

    columns_preprocesses = ['candle_size', 'gap_size', 'high_size', 'low_size', 'volume_size']

    scaler_path = os.path.join(settings.ROOT_DIR, "scalers")

    logging.info("Load Preprocesser Scaler")

    scalers = {}
    for i, tf in enumerate(info):
        scalers[i] = {}
        for j in columns_preprocesses:
            path_lib = "{}/{}/scaler_{}_{}{}.pkl".format(scaler_path, j, j, tf['frame'], tf['type'])
            file = Path(path_lib)
            if file.exists():
                scalers[i][j] = joblib.load(file)
            else:
                logging.error("No se pudo encontrar el Preprocesser Scaler para {} {}".format(tf['type'], tf['frame']))

    input_format = "%Y-%m-%dT%H:%M:%SZ"
    output_format = "%d/%m/%Y"

    START_DATE_SIMULATION = "2010-06-01T00:00:00Z"
    END_DATE_SIMULATION = "2017-10-01T00:00:00Z"

    def env():
        timestamp = random_date(START_DATE_SIMULATION, END_DATE_SIMULATION, input_format,
                                output_format=output_format)
        date = datetime.strptime(timestamp, output_format)

        _start_date = datetime(date.year, date.month, date.day, 6, 0, 0)
        # self.end_date = datetime(date.year, date.month, date.day, 12, 0, 0)
        _end_date = _start_date + timedelta(days=30)

        start_date = _start_date.strftime(settings.DEFAULT_FORMAT)
        end_date = _end_date.strftime(settings.DEFAULT_FORMAT)
        trading_env = TradingEnv(symbol, start_date, end_date, info_st, info_lt, False, False, connection)
        trading_env = RewardMarketRatios(trading_env)
        #trading_env = RewardRiskRatios(trading_env)
        #trading_env = IndicatorBaseWrapper(trading_env)
        #trading_env = BarColorIndicatorWrapper(trading_env)
        if scalers is None:
            trading_env = PreprocessObservation(trading_env, os.path.join(settings.ROOT_DIR, "scalers"))
        else:
            trading_env = PreprocessObservation(trading_env, scalers)
        #trading_env = MainPivotIndicatorWrapper(trading_env)
        return trading_env

    return env


if __name__ == "__main__":
    # Parameters
    n = 1

    symbol = "ES"

    start_date = "2010-06-01 00:00:00"
    end_date = "2010-11-30 23:59:00"

    info_st = {
        'type': 'llc_v3',
        'frame': 3,
        'backperiods': 30,
        'timesteps': 0
    }

    info_lt = [
        {
            'type': 'Min',
            'frame': 15,
            'backperiods': 60,
            'timesteps': 0
        },
        {
            'type': 'Min',
            'frame': 30,
            'backperiods': 60,
            'timesteps': 0
        },
        {
            'type': 'Min',
            'frame': 60,
            'backperiods': 150,
            'timesteps': 0
        },
        {
            'type': 'D',
            'frame': 1,
            'backperiods': 100,
            'timesteps': 0
        }
    ]

    connection = HistoricalConnection(**settings.INFLUXDB_CONNECTION)

    env = create_environment_function_2(symbol, info_st, info_lt, connection)()

    obs = env.reset()

    done = False

    while not done:
        obs, reward, done, info = env.step(1)

        for i, s in enumerate(obs['series']):
            print("Serie: ", i)
            print(s)
