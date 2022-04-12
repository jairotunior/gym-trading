import numpy as np
import time
import math
import pandas as pd
import gym
import gym.spaces
from gym.utils import seeding

from collections import deque

from utils import convert_datetime, get_commisions, resample_ohlc

from datetime import datetime, timedelta

from enum import Enum

from utils.influxdb.connection import HistoricalConnection

import settings

import warnings

import pytz

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from utils import random_date

from gym_trading.render import BitcoinTradingGraph


warnings.simplefilter(action='ignore', category=FutureWarning)

class Actions(Enum):
    Buy = 0
    Sell = 1
    Nothing = 2
    Close = 3
    Moveup_Stoplimit = 4
    Movedown_Stoplimit = 5

class MarketPosition(Enum):
    Long = 0
    Short = 1
    Flat = 2


class State:

    def __init__(self, symbol, start_date, end_date, connection: HistoricalConnection, info_st, info_lt=None, enable_tick_by_tick_step=False, transposed=True):
        assert isinstance(symbol, str)
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)

        self.symbol = symbol
        self.start_date = convert_datetime(start_date)
        self.end_date = convert_datetime(end_date)

        # Convert start_date and end_date to timezone
        tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
        tz_utc = pytz.timezone('UTC')

        print("{} ****** {}".format(start_date, end_date))

        self.start_date = tz_new_york.localize(self.start_date).astimezone(tz_utc)
        self.end_date = tz_new_york.localize(self.end_date).astimezone(tz_utc)

        self.transposed = transposed

        self.info_st = info_st
        self.info_lt = info_lt

        long_term_elements = 0
        if info_lt is not None:
            long_term_elements = len(info_lt)

        self.info = deque(maxlen=long_term_elements + 1)

        # Add Short Term Info
        self.info.append(self.info_st)

        # Add Long Term Info
        if info_lt is not None:
            for i in info_lt:
                self.info.append(i)

        # Tick by Tick Timeserie Enable
        self.enable_tick_by_tick_step =  enable_tick_by_tick_step
        self.switch_tick_by_tick_step = False

        # InfluxDB Connection
        self.connection = connection

        # The market have a screenshot with the essential information for the agent
        self.state = deque(maxlen=len(self.info))

        self.columns = ['open', 'high', 'low', 'close', 'volume', 'cons']
        self.columns_state = ['open', 'high', 'low', 'close', 'volume']

        # BarsArray
        self.bars_array = deque(maxlen=len(self.info))
        self._bars_array = {}
        self.iterator = deque(maxlen=len(self.info))
        self.current_bars = deque(maxlen=len(self.info))

        for i, info in enumerate(self.info):
            self.current_bars.append(info['backperiods'])

            # Validate if resample is needed
            in_list = False
            for ts in settings.INFLUXDB_DEFAULT_SERIES_PER_INSTRUMENT:
                if ts['type'] == info['type'] and ts['frame'] == info['frame']:
                    in_list = True
                    print("Find {} {}".format(ts['type'], ts['frame ']))

            type_query = info['type']
            frame_query = info['frame']

            offset_query = 0

            # ++++++++++++++++++++++++ InfluxDB get data ++++++++++++++++++++++
            # Get the start consecutive start and end
            offset_query = -self.current_bars[-1]

            get_cons = self.connection.get_first_cons(measurement='ohlc', start=self.start_date.strftime("%Y-%m-%d %H:%M:%S"),
                                           end=self.end_date.strftime("%Y-%m-%d %H:%M:%S"), symbol=symbol,
                                           type=type_query, frame=frame_query)

            cons_values = list(get_cons)[0]
            start_cons = cons_values['first']
            end_cons = cons_values['last']

            # Validate if exist enough backperiods
            if start_cons - info['backperiods'] < 0:
                assert IndexError("No se puede manejar indices negativos.")

            # Get the data series with the start_cons and end_cons
            data = self.connection.get_data(start_cons=start_cons, end_cons=end_cons, symbol=symbol, type=type_query, frame=frame_query, offset=offset_query)

            self.bars_array.append(data)

            # El cons de la fila con fecha de inicio de la simulacion
            id_start_date = self.bars_array[-1].loc[start_date:]['cons'][0]
            #print("Start Date: ", self.bars_array[-1].loc[start_date:]['cons'])

            # Truncar bars_array con los datos necesarios solamente, de forma que la fecha de inicio corresponda al numero de backperiods utilizando el iloc
            mask2 = (self.bars_array[-1]['cons'] > (id_start_date - self.current_bars[-1]))
            self.bars_array[-1] = self.bars_array[-1].loc[mask2, self.columns_state]
            #print(self.bars_array[-1].loc[mask2, self.columns_state])

            # Get State --- Option 1
            self.state.append(self.bars_array[-1].iloc[:self.current_bars[-1], :][self.columns_state])

            self.iterator.append(0)


        # Used for syncronize timeframes
        base_bars = [{'type': 'Min', 'frame': 1}]
        for i in base_bars:
            # +++++++++++++++++++++++++++ InfluxDB Get Data ++++++++++++++++++++++++++++++++
            # Get the start consecutive start and end
            cons_values = list(self.connection.get_first_cons(measurement='ohlc', start=self.start_date.strftime("%Y-%m-%d %H:%M:%S"), end=self.end_date.strftime("%Y-%m-%d %H:%M:%S"), symbol=symbol, type=i['type'], frame=i['frame']))[0]
            start_cons = cons_values['first']
            end_cons = cons_values['last']

            self._bars_array[i['type']] = self.connection.get_data(start_cons=start_cons, end_cons=end_cons, symbol=symbol, type=i['type'], frame=i['frame'])


        # Market variables
        self.market_position = MarketPosition.Flat
        self.account_balance = settings.ACCOUNT_MONEY
        self.max_account_balance = self.account_balance
        self.current_price = 0.0
        self.current_date = 0.0
        self.price_entry = 0.0
        self.stoploss = 0.0
        self.stoploss_ticks = 300
        self.target = 0.0
        self.target_ticks = 600

        self.stoploss_position = False
        self.risk_init = 0

        # Current Trade
        self.current_trade = None

        # Trades taken by the agent
        self.data = {
            'date_entry': [],
            'date_exit': [],
            'symbol': [],
            'market_position': [],
            'contracts': [],
            'price_entry': [],
            'price_exit': [],
            'commisions': [],
            'gross_profit': [],
            'profit': []
        }

        # Valor del portafolio
        self.net_worths = [self.account_balance]

        self.trades = pd.DataFrame(data=self.data)

        self.trailing_up_space = ["tu{}".format(index + 1) for index in range(self.stoploss_ticks)]
        self.trailing_up_tick = {t: i for i, t in enumerate(self.trailing_up_space, 1)}

        self.trailing_down_space = ["td{}".format(index + 1) for index in range(self.stoploss_ticks)]
        self.trailing_down_tick = {t: i for i, t in enumerate(self.trailing_down_space, 1)}

        # Space
        self.space = ["buy", "sell", "nothing", "close",
                      *self.trailing_up_space,
                      *self.trailing_down_space]

        self.actions = [['buy'], ['sell'], ['nothing'], ['close'], *[[tu] for tu in self.trailing_up_space], *[[td] for td in self.trailing_down_space]]
        #print("Actions: ", self.actions)
        self._actions = []

        for action in self.actions:
            arr = np.array([0] * len(self.space))
            for a in action:
                arr[self.space.index(a)] = 1
            self._actions.append(arr)

        # Backtesting Metrics
        self.max_drawdown = np.arange(2, dtype=np.float16)
        self.profit_per_month = np.arange(2, dtype=np.float32)

        self.winning_trades = 0
        self.losing_trades = 0
        self.even_trades = 0
        self.net_profit = 0
        self.gross_profit = 0
        self.gross_loss = 0
        self.commisions = 0
        self.average_winning_trades = 0
        self.average_losing_trades = 0
        self.ratio_win_loss = 0
        self.max_consecutive_winners = 0
        self.max_consecutive_loser = 0


    # type = 'tick', frame = '1'
    # Return max, min, last and volume respectively
    def get_last_ticks(self, measurement, symbol, type, frame, last, before_last):
        result = self.connection.query("SELECT price, volume FROM {}.autogen.{} WHERE symbol='{}' and time >= '{}' and time <= '{}'".format(
            "import",
            measurement,
            symbol,
            before_last,
            last
        ))

        df = pd.DataFrame(list(result)[0])

        return df['price'].max(), df['price'].min(), df['price'].last(), df['volume'].sum()

    def get_target_position(self, marketposition, symbol, price_entry, price_exit, contracts):
        assert isinstance(price_entry, float)
        assert isinstance(price_exit, float)
        assert isinstance(marketposition, MarketPosition)
        assert isinstance(symbol, str)
        assert isinstance(contracts, int)

        profit = 0

        if marketposition == MarketPosition.Long:
            # For positive profit the price_exit must be greather than price_entry
            profit = ((price_exit - price_entry) / settings.SYMBOL_LIST[symbol]['tickSize'] *
                      settings.SYMBOL_LIST[symbol]['tickValue']) * contracts
        elif marketposition == MarketPosition.Short:
            # For positive profit the price_exit must be lower then price_entry
            profit = ((price_entry - price_exit) / settings.SYMBOL_LIST[symbol]['tickSize'] *
                      settings.SYMBOL_LIST[symbol]['tickValue']) * contracts

        return profit

    def stack_frames(self, state):
        stack = []
        for i in range(self.stack_elements):
            stack.append(state.iloc[i:self.info[0] + i, :])

        print(stack)

    def sincronize_timeframe(self):
        #print("********************** SINCRONIZE TIMEFRAME ************************")
        for i, info in enumerate(self.info):
            #print("******************* {} ***********************".format(i))

            if i == 0:
                self.state[i] = self.bars_array[i].iloc[self.iterator[i]:self.current_bars[i], :]
                #print(self.state[i])
                continue

            #print(info)
            last_date = self.get_date(i, 0)
            next_date = self.bars_array[i].index[self.current_bars[i]]

            #print("Current Bars Before: ", self.current_bars[i])

            # Step 1: Informacion de las velas formadas
            #print("Date: ", self.current_date)
            #print(self.bars_array[i])
            state = self.bars_array[i].loc[self.bars_array[i].index <= self.current_date]
            #print(state)
            state = state.iloc[-info['backperiods']:, :]

            #print(state.index)
            self.current_bars[i] = self.bars_array[i].index.get_loc(state.index[-1])

            #print("Current Bars After: ", self.current_bars[i])
            #print("STATE")
            #print(state)
            #print("Step 2")
            # Step 2: Verificar si se necesitan velas de 1 minuto
            timeframe_bars = False

            last_time_formation = state.index[-1]

            if (self.current_date - state.index[-1]) > timedelta(minutes=1):
                mask = (self._bars_array['Min'].index > state.index[-1]) & (self._bars_array['Min'].index <= self.current_date)
                formation_bar = self._bars_array['Min'].loc[mask]

                if not formation_bar.empty:
                    #print("Min Formation Bars: True {} **** {}".format(state.index[-1], self.current_date))
                    timeframe_bars = True
                    last_time_formation = formation_bar.index[-1]
                    formation_close = formation_bar.at[last_time_formation, 'close']
                    formation_low = formation_bar.at[last_time_formation, 'low']
                    formation_high = formation_bar.at[last_time_formation, 'high']
                    formation_bar = resample_ohlc(formation_bar, self.info[i]['type'], self.info[i]['frame'])

                    #print("Formation Bar")
                    #print(formation_bar)
                    #as_list = formation_bar.index
                    #as_list[-1] = last_date
                    #formation_bar.index = as_list

                    state = pd.concat([state, formation_bar], axis=0, join='outer', ignore_index=False)

                    if state.shape[0] > info['backperiods']:
                        num_elements = state.shape[0] - info['backperiods']
                        state = state.drop(state.index[[i for i in range(num_elements)]])
            #print(state)
            #print("Step 3")
            # Step 3: Verificar si se necesitan datos Tick by Tick
            if last_time_formation != self.current_date:
                tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
                tz_utc = pytz.timezone('UTC')

                query_start_date = last_time_formation.astimezone(tz_utc)
                query_end_date = self.state[0].index[-1].astimezone(tz_utc)

                #print("Start: {} *** End: {}".format(query_start_date, query_end_date))

                df_ticks = self.connection.get_ticks(query_start_date.strftime("%Y-%m-%d %H:%M:%S.%f"),
                                                     query_end_date.strftime("%Y-%m-%d %H:%M:%S.%f"), self.symbol)
                #print("TICKS")
                #print(df_ticks)

                if df_ticks is not None:
                    if not timeframe_bars:
                        formation_bar = df_ticks['price'].resample('{}{}'.format(info['frame'], info['type'])).ohlc()
                        vol = df_ticks['volume'].sum()
                        formation_bar['volume'] = vol
                        #formation_bar.index =

                        #print("Formation Bar 1")
                        #print(formation_bar)

                        state = pd.concat([state, formation_bar], axis=0, join='outer', ignore_index=False)

                        if state.shape[0] > self.info[i]['backperiods']:
                            num_elements = state.shape[0] - info['backperiods']
                            state = state.drop(state.index[[i for i in range(num_elements)]])

                        #print(state)
                        #print("Volumen: ", vol)
                    else:
                        #print("Formation Bar 2")
                        state.iloc[-1, 3] = df_ticks.iloc[-1, :]['price']
                        tick_high = df_ticks['price'].max()
                        tick_low = df_ticks['price'].min()
                        vol = df_ticks['volume'].sum()

                        close_candle_time = state.index[-1]
                        state.at[close_candle_time, 'volume'] += vol

                        if tick_high > state.iloc[-1, 1]:
                            state.iat[-1, 1] = tick_high
                        if tick_low < state.iloc[-1, 2]:
                            state.iat[-1, 2] = tick_low

                        #print(state)
                        #print(vol)
                else:
                    #print("Formation Bar 3")
                    close_base = self.state[0].iloc[-1, :]['close']
                    high_base = self.state[0].iloc[-1, :]['high']
                    low_base = self.state[0].iloc[-1, :]['low']

                    # Close LT Candle must be equal to LT Candle
                    state.iat[-1, 3] = close_base

            """
            # Si la fecha de cierre de la vela actual (LT) es mayor a la fecha de cierre de la vela actual (ST)
            # Entonces se debe crear una vela en formacion
            if self.current_date < next_date and self.current_date > last_date:

                # Se debe crear una vela en formacion
                mask = (self._bars_array['Min'].index > last_date) & (self._bars_array['Min'].index <= self.current_date)
                formation_bar = self._bars_array['Min'].loc[mask]

                print("Range Dates: {} **** {} ********* {}".format(last_date, self.current_date, self.state[0].index[-1]))
                print("Formation Bar 1")

                join = state

                if not formation_bar.empty:
                    last_time_formation = formation_bar.index[-1]
                    formation_close = formation_bar.at[last_time_formation, 'close']
                    formation_low = formation_bar.at[last_time_formation, 'low']
                    formation_high = formation_bar.at[last_time_formation, 'high']

                    formation_bar = resample_ohlc(formation_bar, self.info[i]['type'], self.info[i]['frame'])

                    close_candle_time = formation_bar.index[-1]

                    join = pd.concat([state, formation_bar], axis=0, join='outer', ignore_index=False)

                    if join.shape[0] > self.info[i]['backperiods']:
                        num_elements = join.shape[0] - self.info[i]['backperiods']
                        join = join.drop(join.index[[i for i in range(num_elements)]])
                        # join = join.dropna(axis=0)
                else:
                    last_time_formation = last_date
                    close_candle_time = last_date


                # Verify if need to get tick by tick data
                if last_time_formation != self.state[0].index[-1]:
                    print("Fecha Velas Formadas: {} - Fecha ultima vela: {}".format(last_time_formation, self.state[0].index[-1]))

                    tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
                    tz_utc = pytz.timezone('UTC')

                    query_start_date = last_time_formation.astimezone(tz_utc)
                    query_end_date = self.state[0].index[-1].astimezone(tz_utc)

                    df_ticks = self.connection.get_ticks(query_start_date.strftime("%Y-%m-%d %H:%M:%S.%f"), query_end_date.strftime("%Y-%m-%d %H:%M:%S.%f"), self.symbol)
                    #print("TICKS")
                    #print(df_ticks)


                    if df_ticks is not None:
                        join.iloc[-1, 3] = df_ticks.iloc[-1, :]['price']
                        tick_high = df_ticks['price'].max()
                        tick_low = df_ticks['price'].min()
                        vol = df_ticks['volume'].sum()

                        #print("CASE 1")
                        #print(join[self.columns_state])
                        #print("Volume in Tick: {}".format(vol))

                        join.at[close_candle_time, 'volume'] += vol

                        if tick_high > join.iloc[-1, 1]:
                            join.iat[-1, 1] = tick_high
                        if tick_low < join.iloc[-1, 2]:
                            join.iat[-1, 2] = tick_low
                        #print(join[self.columns_state])
                    else:
                        close_base = self.state[0].iloc[-1, :]['close']
                        high_base = self.state[0].iloc[-1, :]['high']
                        low_base = self.state[0].iloc[-1, :]['low']

                        #print("CASE 2")
                        #print(join[self.columns_state])
                        #print("******************")

                        # Close LT Candle must be equal to LT Candle
                        join.iat[-1, 3] = close_base

                        #print(join[self.columns_state])

                self.state[i] = join[self.columns_state]

            elif self.current_date > next_date:

                iterator = self.iterator[i]
                current_bar = self.current_bars[i]

                while self.current_date < next_date:
                    iterator += 1
                    current_bar += 1

                    next_date = self.bars_array[i].index[current_bar]

                self.iterator[i] = iterator
                self.current_bars[i] = current_bar

                last_date = self.bars_array[i].index[self.current_bars[i] - 1]


                mask = (self._bars_array['Min'].index > last_date) & (self._bars_array['Min'].index <= self.current_date)
                formation_bar = self._bars_array['Min'].loc[mask]

                print("Range Dates: {} **** {} ******** {}".format(last_date, self.current_date, self.state[0].index[-1]))
                print("Formation Bar 2")

                join = state

                if not formation_bar.empty:
                    last_time_formation = formation_bar.index[-1]
                    formation_close = formation_bar.at[last_time_formation, 'close']
                    formation_low = formation_bar.at[last_time_formation, 'low']
                    formation_high = formation_bar.at[last_time_formation, 'high']

                    formation_bar = resample_ohlc(formation_bar, self.info[i]['type'], self.info[i]['frame'])

                    close_candle_time = formation_bar.index[-1]

                    join = pd.concat([state, formation_bar], axis=0, join='outer', ignore_index=False)

                    if join.shape[0] > self.info[i]['backperiods']:
                        num_elements = join.shape[0] - self.info[i]['backperiods']
                        join = join.drop(join.index[[i for i in range(num_elements)]])
                else:
                    last_time_formation = last_date
                    close_candle_time = last_date


                # Verify if need to get tick by tick data
                if last_time_formation != self.state[0].index[-1]:
                    print("Fecha Velas Formadas: {} - Fecha ultima vela: {}".format(last_time_formation, self.state[0].index[-1]))
                    #print("{} ****** {}".format(last_time_formation.strftime("%Y-%m-%d %H:%M:%S.%f"), self.state[0].index[-1].strftime("%Y-%m-%d %H:%M:%S.%f")))

                    tz_new_york = pytz.timezone(settings.DEFAULT_TIME_ZONE)
                    tz_utc = pytz.timezone('UTC')

                    query_start_date = last_time_formation.astimezone(tz_utc)
                    query_end_date = self.state[0].index[-1].astimezone(tz_utc)

                    df_ticks = self.connection.get_ticks(query_start_date.strftime("%Y-%m-%d %H:%M:%S.%f"), query_end_date.strftime("%Y-%m-%d %H:%M:%S.%f"), self.symbol)
                    #print("TICKS")
                    #print(df_ticks)


                    if df_ticks is not None:
                        join.iloc[-1, 3] = df_ticks.iloc[-1, :]['price']
                        tick_high = df_ticks['price'].max()
                        tick_low = df_ticks['price'].min()
                        vol = df_ticks['volume'].sum()

                        #print("CASE 3")
                        #print(join[self.columns_state])
                        #print("Volume in Tick: {}".format(vol))

                        join.at[close_candle_time, 'volume'] += vol

                        if tick_high > join.iloc[-1, 1]:
                            join.iat[-1, 1] = tick_high
                        if tick_low < join.iloc[-1, 2]:
                            join.iat[-1, 2] = tick_low

                        #print(join[self.columns_state])
                    else:
                        close_base = self.state[0].iloc[-1, :]['close']
                        high_base = self.state[0].iloc[-1, :]['high']
                        low_base = self.state[0].iloc[-1, :]['low']

                        #print("CASE 4")
                        #print(join[self.columns_state])
                        #print("******************")
                        # Close LT Candle must be equal to LT Candle
                        join.iat[-1, 3] = close_base

                        #print(join[self.columns_state])

                self.state[i] = join[self.columns_state]

            # El caso en que la fecha de la vela actual (ST) es igual a la siguiente vela (LT)
            else:
                self.iterator[i] += 1
                self.current_bars[i] += 1

                state = self.bars_array[i].iloc[self.iterator[i]:self.current_bars[i], :]

                #print("Formation Bar 3: {}".format(state))

                self.state[i] = state
            """

    def step(self, action):
        self.iterator[0] += 1
        self.current_bars[0] += 1

        self.current_price = self.get_close(0, 0)
        self.current_date = self.get_date(0, 0)
        self.sincronize_timeframe()

        self.stoploss_position = False

        reward = 0.0

        if self.market_position == MarketPosition.Flat:
            if self.space[action] == "buy":
                self.market_position = MarketPosition.Long
                self.price_entry = self.current_price
                self.stoploss = self.price_entry - round(self.stoploss_ticks) * settings.SYMBOL_LIST[self.symbol]['tickSize']
                self.target = self.price_entry + round(self.target_ticks) * settings.SYMBOL_LIST[self.symbol]['tickSize']

                # Initial Risk
                self.risk_init = abs(self.price_entry - self.stoploss)

                # Reward
                reward -= get_commisions(self.symbol, settings.CONTRACTS)

                # Register trade information
                self.current_trade = {}
                self.current_trade['date_entry'] = self.get_date(0, 0)
                self.current_trade['symbol'] = self.symbol
                self.current_trade['market_position'] = MarketPosition.Long
                self.current_trade['contracts'] = settings.CONTRACTS
                self.current_trade['price_entry'] = self.current_price
                self.current_trade['commisions'] = reward
                self.current_trade['gross_profit'] = 0
                self.current_trade['profit'] = 0
                self.current_trade['price_exit'] = 0
                self.current_trade['date_exit'] = 0
            elif self.space[action] == "sell":
                self.market_position = MarketPosition.Short
                self.price_entry = self.current_price
                self.stoploss = self.price_entry + round(self.stoploss_ticks) * settings.SYMBOL_LIST[self.symbol]['tickSize']
                self.target = self.price_entry - round(self.target_ticks) * settings.SYMBOL_LIST[self.symbol]['tickSize']

                # Initial Risk
                self.risk_init = abs(self.price_entry - self.stoploss)

                # Reward
                reward -= get_commisions(self.symbol, settings.CONTRACTS)

                # Register trade information
                self.current_trade = {}
                self.current_trade['date_entry'] = self.get_date(0, 0)
                self.current_trade['symbol'] = self.symbol
                self.current_trade['market_position'] = MarketPosition.Short
                self.current_trade['contracts'] = settings.CONTRACTS
                self.current_trade['price_entry'] = self.current_price
                self.current_trade['commisions'] = reward
                self.current_trade['gross_profit'] = 0
                self.current_trade['profit'] = 0
                self.current_trade['price_exit'] = 0
                self.current_trade['date_exit'] = 0
        elif self.market_position != MarketPosition.Flat:
            # Close position by Stoploss
            if self.stoploss <= self.get_high(0, 0) and self.stoploss >= self.get_low(0, 0):
                print("Stoploss")
                # Register the close_position trade
                profit = self.close_position(self.get_date(0, 0), self.stoploss)
                self.trades = self.trades.append(self.current_trade, ignore_index=True)

                reward -= get_commisions(self.current_trade['symbol'], self.current_trade['contracts'])
                reward += self.get_target_position(self.market_position, self.symbol, self.get_close(0, 1),
                                                   self.stoploss, self.current_trade['contracts'])

                # Update the account balance
                self.account_balance += profit

                # Update maximum account balance
                if self.max_account_balance < self.account_balance:
                    self.max_account_balance = self.account_balance

                # The position was stoploss
                self.stoploss_position = True

                # Reset market to Flat
                self.market_position = MarketPosition.Flat
                self.stoploss = 0.0
                self.target = 0.0
                self.contracts = 0
                self.price_entry = 0.0

                self.risk_init = 0
                self.current_trade = None

            # Close position by Target
            elif (self.market_position == MarketPosition.Long and self.target <= self.get_high(0, 0)) or \
                    (self.market_position == MarketPosition.Short and self.target >= self.get_low(0, 0)):
                print("Target - High: ", self.get_high(0, 0), " - Low: ", self.get_low(0, 0))
                # Register the close_position trade
                profit = self.close_position(self.get_date(0, 0), self.target)
                self.trades = self.trades.append(self.current_trade, ignore_index=True)

                reward -= get_commisions(self.current_trade['symbol'], self.current_trade['contracts'])
                reward += self.get_target_position(self.market_position, self.symbol, self.get_close(0, 1),
                                                   self.stoploss, self.current_trade['contracts'])

                # Update the account balance
                self.account_balance += profit

                # Update maximum account balance
                if self.max_account_balance < self.account_balance:
                    self.max_account_balance = self.account_balance

                # Reset market to Flat
                self.market_position = MarketPosition.Flat
                self.stoploss = 0.0
                self.target = 0.0
                self.contracts = 0
                self.price_entry = 0.0

                self.risk_init = 0
                self.current_trade = None

            elif self.space[action] == "nothing":
                if (self.stoploss < self.get_high(0, 0) and self.stoploss > self.get_low(0, 0)) or (
                        self.stoploss < self.get_high(0, 0) and self.stoploss < self.get_low(0, 0)):
                    reward += self.get_target_position(self.current_trade['market_position'],
                                                       self.current_trade['symbol'],
                                                       self.get_close(0, 1),
                                                       self.stoploss,
                                                       self.current_trade['contracts'])
                else:
                    reward += self.get_target_position(self.current_trade['market_position'],
                                                       self.current_trade['symbol'],
                                                       self.get_close(0, 1),
                                                       self.get_close(0, 0),
                                                       self.current_trade['contracts'])
            elif self.space[action] in self.trailing_up_space and self.market_position == MarketPosition.Long:
                # self.stoploss += round(abs(self.current_price - self.stoploss - 1) / settings.SYMBOL_LIST[self.symbol]['tickSize'] * self.trailing_up_tick[self.space[action]]) * settings.SYMBOL_LIST[self.symbol]['tickSize']
                self.stoploss += settings.SYMBOL_LIST[self.symbol]['tickSize'] * self.trailing_up_tick[self.space[action]]
            elif self.space[action] in self.trailing_down_space and self.market_position == MarketPosition.Short:
                # self.stoploss -= round(abs(self.current_price - self.stoploss - 1) / settings.SYMBOL_LIST[self.symbol]['tickSize'] * self.trailing_down_tick[self.space[action]]) * settings.SYMBOL_LIST[self.symbol]['tickSize']
                self.stoploss -= self.trailing_down_tick[self.space[action]] * settings.SYMBOL_LIST[self.symbol]['tickSize']
            elif self.space[action] == "close":
                # Register the close_position trade
                profit = self.close_position(self.get_date(0, 0), self.get_close(0, 0))
                self.trades = self.trades.append(self.current_trade, ignore_index=True)

                reward -= get_commisions(self.current_trade['symbol'], self.current_trade['contracts'])
                # reward += self.get_target_position(self.market_position, self.symbol, self.price_entry, self.current_price, contracts)
                reward += self.get_target_position(self.market_position, self.symbol, self.get_close(0, 1),
                                                   self.get_close(0, 0), self.current_trade['contracts'])

                # Update the account balance
                self.account_balance += profit

                # Update maximum account balance
                if self.max_account_balance < self.account_balance:
                    self.max_account_balance = self.account_balance

                # Reset market to Flat
                self.market_position = MarketPosition.Flat
                self.stoploss = 0.0
                self.contracts = 0
                self.price_entry = 0.0

                self.risk_init = 0
                self.current_trade = None


        # Save the current portafolio value
        self.net_worths.append(self.account_balance + reward)

        #done = self._dones()
        done = self._dones_end_simulation()

        return self.get_state(), reward, done

    def reset(self):
        self.market_position = MarketPosition.Flat

        self.account_balance = settings.ACCOUNT_MONEY
        self.max_account_balance = self.account_balance

        # Add account balance to net worths list
        self.net_worths = [self.account_balance]

        for i, info in enumerate(self.info):
            self.iterator[i] = 0
            self.current_bars[i] = info['backperiods']

        self.current_date = self.get_date(0, 0)
        self.current_price = self.get_close(0, 0)

        self.stoploss = 0.0
        self.target = 0.0

        # Sincroniza las series de tiempo
        self.sincronize_timeframe()

        self.trades = pd.DataFrame(data=self.data)
        self.current_trade = None

    def _dones_maximum_loss(self):
        maximum_loss = self.net_worths[-1] < self.max_account_balance * 0.9
        return maximum_loss

    def _dones_end_simulation(self):
        end_simulation = [False if self.bars_array[0].shape[0] > self.current_bars[0] else True][0]
        return end_simulation

    def _dones(self):
        maximum_loss = self._dones_maximum_loss()
        end_simulation = self._dones_end_simulation()

        result = 0
        if maximum_loss:
            result = 1
        elif end_simulation:
            result = 2

        #return maximum_loss or end_simulation
        return result


    def close_position(self, date_exit, price_exit):
        self.current_trade['date_exit'] = date_exit
        self.current_trade['price_exit'] = price_exit
        self.current_trade['commisions'] -= get_commisions(self.current_trade['symbol'], self.current_trade['contracts'])
        self.current_trade['gross_profit'] += self.get_target_position(self.current_trade['market_position'], self.current_trade['symbol'],
                                                                  self.current_trade['price_entry'], price_exit, self.current_trade['contracts'])
        self.current_trade['profit'] = self.current_trade['gross_profit'] + self.current_trade['commisions']

        return self.current_trade['profit']

    # Return an array of DataFrame
    def get_state(self):
        return [s for s in self.state]
        #return [bar.iloc[self.iterator[i]:self.current_bars[i], :] for i, bar in enumerate(self.bars_array)]

    def get_state_shape(self):
        if self.transposed:
            return [s.transpose().shape for s in self.state]
        return [s.shape for s in self.state]

    def get_contracts(self, risk, price_entry, stoploss):
        tick_size = settings.SYMBOL_LIST[self.symbol]['tickSize']
        tick_value = settings.SYMBOL_LIST[self.symbol]['tickValue']

        return math.trunc(risk / ((price_entry - stoploss)/tick_size) * tick_value)

    def reward_risk(self):
        risk = self.current_price - self.stoploss
        reward = self.current_price - self.price_entry

        if self.market_position == MarketPosition.Short:
            risk = self.stoploss - self.current_price
            reward = self.price_entry - self.current_price

        return reward/risk

    def reward_risk_management(self):
        risk, reward = [0,0]
        if self.market_position == MarketPosition.Long:
            risk = self.price_entry - self.stoploss
            reward = self.current_price - self.price_entry
        elif self.market_position == MarketPosition.Short:
            risk = self.stoploss - self.price_entry
            reward = self.price_entry - self.current_price

        if reward == float(0):
            return 0

        return risk/reward

    def reward_large_trades(self):
        risk = abs(self.price_entry - self.stoploss)

        if self.market_position == MarketPosition.Long:
            reward = self.current_price - self.price_entry
        elif self.market_position == MarketPosition.Short:
            reward = self.price_entry - self.current_price

        return reward / risk


    #region ********************************* Bars Info *********************************************
    """
        This methods only return the information of the formed bar. These methods doesn't take a count the bar in formation.
    """
    def get_date(self, bar_index=0, index=0):
        #print(self.bars_array[bar_index])
        return self.bars_array[bar_index].index[self.current_bars[bar_index] - index - 1]

    def get_open(self, bar_index=0, index=0):
        return self.bars_array[bar_index].iloc[self.current_bars[bar_index] - index - 1, :]['open']

    def get_high(self, bar_index=0, index=0):
        return self.bars_array[bar_index].iloc[self.current_bars[bar_index] - index - 1, :]['high']

    def get_low(self, bar_index=0, index=0):
        return self.bars_array[bar_index].iloc[self.current_bars[bar_index] - index - 1, :]['low']

    def get_close(self, bar_index=0, index=0):
        return self.bars_array[bar_index].iloc[self.current_bars[bar_index] - index - 1, :]['close']

    #endregion

    # region ********************************** Backtesting Metrics **********************************

    def get_winning_trades(self, market_position=None):
        if market_position is None:
            mask = (self.trades['profit'] > 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
        return self.trades[mask]['profit'].count()

    def get_losing_trades(self, market_position=None):
        if market_position is None:
            mask = (self.trades['profit'] < 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
        return self.trades[mask]['profit'].count()

    def get_even_trades(self, market_position=None):
        if market_position is None:
            mask = (self.trades['profit'] == 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] == 0)
        return self.trades[mask]['profit'].count()

    def get_net_profit(self, market_position=None):
        if market_position is None:
            return self.trades['profit'].sum()
        else:
            mask = (self.trades['market_position'] == market_position)

        return self.trades[mask]['profit'].sum()

    def get_gross_profit(self, market_position=None):
        if market_position is None:
            mask = (self.trades['gross_profit'] > 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['gross_profit'] > 0)
        return self.trades[mask]['gross_profit'].sum()

    def get_gross_loss(self, market_position=None):
        if market_position is None:
            mask = (self.trades['gross_profit'] < 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['gross_profit'] < 0)
        return self.trades[mask]['gross_profit'].sum()

    def get_commisions(self, market_position=None):
        if market_position is None:
            return self.trades['commisions'].sum()

        mask = (self.trades['market_position'] == market_position)
        return self.trades[mask]['commisions'].sum()

    def get_average_winning_trades(self, market_position=None):
        if market_position is None:
            mask = (self.trades['profit'] > 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
        return self.trades[mask]['profit'].mean()

    def get_average_losing_trades(self, market_position=None):
        if market_position is None:
            mask = (self.trades['profit'] < 0)
        else:
            mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
        return self.trades[mask]['profit'].mean()

    def get_ratio_win_loss(self, market_position=None):
        return np.abs(self.get_average_winning_trades(market_position) / self.get_average_losing_trades(market_position))

    def get_max_consecutive_winners(self, market_position=None):
        self.trades['consecutive'] = np.sign(self.trades['profit'])
        self.trades['count_consecutive'] = self.trades['consecutive'] * (self.trades['consecutive'].groupby(
            (self.trades['consecutive'] != self.trades['consecutive'].shift()).cumsum()).cumcount() + 1)
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
        return self.trades[mask]['count_consecutive'].max()

    def get_max_consecutive_loser(self, market_position=None):
        self.trades['consecutive'] = np.sign(self.trades['profit'])
        self.trades['count_consecutive'] = self.trades['consecutive'] * (self.trades['consecutive'].groupby(
            (self.trades['consecutive'] != self.trades['consecutive'].shift()).cumsum()).cumcount() + 1)

        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
        return self.trades[mask]['count_consecutive'].max()

    # endregion


    @property
    def total_profit(self):
        return self.trades['profit'].sum()

    @property
    def shape(self):
        return [i.shape for i in self.state]


class TradingEnvSup(gym.Env):

    metadata = {'render.modes': ['human', 'system']}

    def __init__(self, symbol=None, start_date=None, end_date=None, info_st=None, info_lt=None, enable_tick_by_tick_step=False, use_settings=True, connection: HistoricalConnection=None, transposed=True, jump_date=False):

        if use_settings:
            assert getattr(settings, "SYMBOL", False), "The setting file don't contain SYMBOL attribute"
            assert getattr(settings, "START_DATE", False), "The setting file don't contain START_DATE attribute"
            assert getattr(settings, "END_DATE", False), "The setting file don't contain END_DATE attribute"
            assert getattr(settings, "FREQ_SHORT_TERM", False), "The setting file don't contain FREQ_LONG_TERM attribute"

            self.symbol = settings.SYMBOL
            self.start_date = settings.START_DATE
            self.end_date = settings.END_DATE
            self.info_st = settings.FREQ_SHORT_TERM
            self.enable_tick_by_tick_step = settings.ENABLE_TICK_BY_TICK_STEP

            if hasattr(settings, "FREQ_LONG_TERM"):
                self.info_lt = settings.FREQ_LONG_TERM

        assert symbol in settings.SYMBOL_LIST, 'The symbol name is not registered in settings file.'
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        assert isinstance(info_st, dict)

        if info_lt is not None:
            assert isinstance(info_lt, list), "The parameter freq_lt must be an array"

        if connection is not None:
            assert isinstance(connection, HistoricalConnection), "The parameter connection must be an HistoricalConnection object."

        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date,
        self.info_st = info_st
        self.info_lt = info_lt
        self.enable_tick_by_tick_step = enable_tick_by_tick_step
        self.connection = connection
        self.transposed = transposed
        self.jump_date = jump_date

        self.current_reward = 0

        self.last_action = None

        self.market = State(symbol=self.symbol, start_date=start_date, end_date=end_date, info_st=info_st, info_lt=info_lt,
                            connection=self.connection, enable_tick_by_tick_step=self.enable_tick_by_tick_step, transposed=self.transposed)

        self.action_space = gym.spaces.Discrete(n=len(self.market._actions)) # Discrete Actions: BUY, SELL, NOTHING, CLOSE, TRAILING STOP FOR LONG POSITION AND SHORT POSITION

        # self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self.market.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
                                        'series': gym.spaces.Tuple(tuple([
                                                                    gym.spaces.Box(low=0, high=1, shape=i + (1,), dtype=np.float32)
                                                                 for i in self.market.get_state_shape()])),
        })

        self.action_token = Tokenizer(self.action_space.n)
        self.action_token.fit_on_texts(self.market.space)

        self.rewards = []

        self.seed()

        self._market_position_discretizer()

        # Definir intervalo de fechas de simulacion
        self.START_DATE_SIMULATION = "2010-06-01T00:00:00Z"
        #self.START_DATE_SIMULATION = list(self.connection.query("SELECT FIRST(open) FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}'".format(self.connection.database, 'ohlc', self.symbol, 'Min', 1)))[0]
        #self.END_DATE_SIMULATION = list(self.connection.query("SELECT LAST(open) FROM {}.autogen.{} WHERE symbol='{}' and type='{}' and frame='{}'".format(self.connection.database, 'ohlc', self.symbol, 'Min', 1)))[0]
        self.END_DATE_SIMULATION = "2017-10-01T00:00:00Z"

        self.START_DATE_SIMULATION  = "2010-06-01T00:00:00Z"
        self.END_DATE_SIMULATION = "2017-10-01T00:00:00Z"
        #self.END_DATE_SIMULATION  = list(self.END_DATE_SIMULATION)[0]['time']

        self.last_date = self.market.start_date


    def _market_position_discretizer(self):
        market_positions = ['Long', 'Short', 'Flat']
        self._market_positions = []

        for i, m in enumerate(market_positions):
            arr = np.array([0]*len(MarketPosition))
            arr[i] = 1
            self._market_positions.append(arr)

    def get_market_position_discrete(self):
        return self._market_positions[self.market.market_position.value]

    def get_available_actions_names(self):
        actions = []

        if self.market.market_position == MarketPosition.Flat:
            actions = ['buy', 'sell', 'nothing']
        elif self.market.market_position == MarketPosition.Long:
            ticks = int((self.market.current_price - self.market.stoploss - settings.SYMBOL_LIST[self.symbol]['tickSize']) / settings.SYMBOL_LIST[self.symbol]['tickSize'])
            actions = ["nothing", "close", *["tu{}".format(index + 1) for index in range(min(12, ticks))]]
        elif self.market.market_position == MarketPosition.Short:
            ticks = int((self.market.stoploss - self.market.current_price - settings.SYMBOL_LIST[self.symbol]['tickSize']) / settings.SYMBOL_LIST[self.symbol]['tickSize'])
            actions = ["nothing", "close", *["td{}".format(index + 1) for index in range(min(12, ticks))]]

        return actions

    def get_stoploss_ticks(self):
        if self.market.market_position != MarketPosition.Flat:
            return abs(self.market.current_price - self.market.stoploss)
        return 0

    def step(self, action):
        #print("++++++++++++++++++++++ INIT +++++++++++++++++++++++++")
        #print("Action: {} - {}".format(action, self.market.actions[action]))
        #print("********************************************************")
        #name_action = self.market.space[action]
        #print("Action: ", name_action)

        self.last_action = action
        new_state, self.current_reward, done = self.market.step(action)

        gameover = 1 if done > 0 else 0
        done = True if done > 0 else False

        self.rewards.append(self.current_reward)
        info = {
            'reward': self.current_reward,
            'balance': self.market.account_balance,
            #'length': len(self.rewards),
            'gameover': gameover,
            #'num_days': int(time.mktime(self.market.current_date.timetuple()) - time.mktime(self.market.start_date.timetuple())) / 60
            'num_days': (self.market.current_date - self.last_date).total_seconds() / 86400
        }

        # Last date
        self.last_date = self.market.current_date

        stoploss = self.get_stoploss_ticks()
        #print("Stoploss: ", stoploss)

        obs = {
            'series': new_state
        }

        #return obs, self._reward(self.market.space[action]), done, info
        return obs, 0, done, info


    def reset(self):
        if not self.jump_date:
            self.market.reset()

            self.last_date = self.market.start_date

            self.rewards = []

            obs = {
                'series': self.market.get_state(),
            }

            return obs

        return self._recreate()


    def _recreate(self):
        input_format = "%Y-%m-%dT%H:%M:%SZ"
        output_format = "%d/%m/%Y"

        timestamp = random_date(self.START_DATE_SIMULATION, self.END_DATE_SIMULATION, input_format, output_format=output_format)
        date = datetime.strptime(timestamp, output_format)

        self.start_date = datetime(date.year, date.month, date.day, 6, 0, 0)
        #self.end_date = datetime(date.year, date.month, date.day, 12, 0, 0)
        self.end_date = self.start_date + timedelta(days=30)

        self.market = State(symbol=self.symbol, start_date=self.start_date.strftime(settings.DEFAULT_FORMAT), end_date=self.end_date.strftime(settings.DEFAULT_FORMAT), info_st=self.info_st,
                            info_lt=self.info_lt, connection=self.connection,
                            enable_tick_by_tick_step=self.enable_tick_by_tick_step)

        self.last_date = self.market.start_date

        self.rewards = []

        obs = {
            'series': self.market.get_state(),
            'stoploss': np.array(self.get_stoploss_ticks()),
            'market_position': self.get_market_position_discrete(),
            'available_actions': self.get_available_actions_embedding(),
            'available_actions_one_hot': self.get_available_actions_one_hot(),
        }

        return obs


    def close(self):
        pass

    def _reward(self, action):
        risk = 0
        profit = 0
        _reward = 0

        if self.market.market_position != MarketPosition.Flat:
            risk = self.market.price_entry - self.market.stoploss + 1
            profit = self.market.current_price - self.market.price_entry

            # No hay riesgo. El stoploss esta por encima del breakeven
            if risk < 0:
                if profit > 0:
                    print("*1Numerator: {} - Denominator: {}".format(profit + self.market.risk_init,
                                                                    profit + self.market.risk_init + risk))
                    print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
                    _reward = ((profit + self.market.risk_init) / (profit + self.market.risk_init + risk)) * 5
                else:
                    print("*2Numerator: {} - Denominator: {}".format(self.market.risk_init,
                                                                    self.market.risk_init + risk))
                    print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
                    _reward = (self.market.risk_init) / (self.market.risk_init + risk)
                    if _reward == float('inf'):
                        _reward = 0
            # El stoploss esta por debajo del breakeven
            elif risk > 0:
                print("**Numerator: {} - Denominator: {}".format(profit + self.market.risk_init, risk))
                print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
                _reward = (profit + self.market.risk_init) / risk
            # El stoploss esta en Breakeven
            else:
                print("***Numerator: {} - Denominator: {}".format(0, 0))
                print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
                _reward = 20

        if self.market.space[self.last_action] == "close":
            print("****CLOSE POSITION")
            print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
            _reward = self.market.account_balance / settings.ACCOUNT_MONEY
        """
        elif self.env.market.market_position == MarketPosition.Flat:
            if self.env.market.stoploss_position:
                _reward = -1
        """

        if self.market._dones_maximum_loss():
            print("*****MAXIMUM LOSS")
            print("Market Position: {} *** Current Price:{} *** Price Entry: {} *** Stoploss: {}".format(self.market.market_position, self.market.current_price, self.market.price_entry, self.market.stoploss))
            _reward = -100

        print("Profit: {} -  Risk Init: {} -  Risk: {} - Reward: {}".format(
            profit, self.market.risk_init, risk, _reward))
        print("Action: {}".format(action))

        if _reward == float("inf"):
            _reward = 0

        self.current_reward = _reward

        return _reward

    def render(self, mode='human'):
        #self._render_system()

        tick_size = settings.SYMBOL_LIST[self.symbol]['tickSize']
        tick_value = settings.SYMBOL_LIST[self.symbol]['tickValue']

        profit_loss = 0

        if self.market.market_position == MarketPosition.Long:
            profit_loss = ((self.market.current_price - self.market.current_trade['price_entry']) / tick_size) * \
                          self.market.current_trade['contracts'] * tick_value
        elif self.market.market_position == MarketPosition.Short:
            profit_loss = ((self.market.current_trade['price_entry'] - self.market.current_price) / tick_size) * \
                          self.market.current_trade['contracts'] * tick_value

        print("{:*^80}".format(" SUMMARY "))
        print("*** Datetime: {}".format(self.market.current_date))
        print("*** Market Position: {}".format(self.market.market_position))
        print("*** Last Price: {}".format(self.market.current_price))
        print("*** Action: {}".format(self.last_action))
        print("*** Account Balance: {0:.2f} *** Profit / Loss: {0:.2f}".format(self.market.account_balance, profit_loss))
        print("*** Account Balance + P/L: {0:.2f}".format(self.market.account_balance + profit_loss))
        print("*** Reward Function: {0:.2f}".format(self.current_reward))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("*** {:<20} {:>12} {:>12} {:>12}".format("PERFORMANCE", "TOTAL", "LONG", "SHORT"))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Total Net Profit",
                                                       "{0:.2f}".format(self.market.get_net_profit()),
                                                       "{0:.2f}".format(self.market.get_net_profit(MarketPosition.Long)),
                                                       "{0:.2f}".format(self.market.get_net_profit(MarketPosition.Short))))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Gross Profit",
                                                       "{0:.2f}".format(self.market.get_gross_profit()),
                                                       "{0:.2f}".format(self.market.get_gross_profit(MarketPosition.Long)),
                                                       "{0:.2f}".format(self.market.get_gross_profit(MarketPosition.Short))))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Gross Loss",
                                                       "{0:.2f}".format(self.market.get_gross_loss()),
                                                       "{0:.2f}".format(self.market.get_gross_loss(MarketPosition.Long)),
                                                       "{0:.2f}".format(self.market.get_gross_loss(MarketPosition.Short))))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Commisions",
                                                       "{0:.2f}".format(self.market.get_commisions()),
                                                       "{0:.2f}".format(self.market.get_commisions(MarketPosition.Long)),
                                                       "{0:.2f}".format(self.market.get_commisions(MarketPosition.Short))))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Winning Trades",
                                                       self.market.get_winning_trades(),
                                                       self.market.get_winning_trades(MarketPosition.Long),
                                                       self.market.get_winning_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Losing Trades",
                                                       self.market.get_losing_trades(),
                                                       self.market.get_losing_trades(MarketPosition.Long),
                                                       self.market.get_losing_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Even Trades",
                                                       self.market.get_even_trades(),
                                                       self.market.get_even_trades(MarketPosition.Long),
                                                       self.market.get_even_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Avg. Winning Trades",
                                                                self.market.get_average_winning_trades(),
                                                                self.market.get_average_winning_trades(MarketPosition.Long),
                                                                self.market.get_average_winning_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Avg. Losing Trades",
                                                                self.market.get_average_losing_trades(),
                                                                self.market.get_average_losing_trades(MarketPosition.Long),
                                                                self.market.get_average_losing_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Ratio Win/Loss",
                                                                self.market.get_ratio_win_loss(),
                                                                self.market.get_ratio_win_loss(MarketPosition.Long),
                                                                self.market.get_ratio_win_loss(MarketPosition.Short)))

        if self.market.market_position != MarketPosition.Flat:
            print("--------------------------TRADES--------------------------------")
            print("*** Datetime: {}".format(self.market.current_trade['date_entry']))
            print("*** Contracts: {}".format(self.market.current_trade['contracts']))
            print("*** Symbol: {}".format(self.market.current_trade['symbol']))
            print("*** Price Entry: {}".format(self.market.current_trade['price_entry']))
            print("*** Price Exit: {}".format(self.market.current_trade['price_exit']))
            print("*** Stoploss: {}".format(self.market.stoploss))
            print("*** Target: {}".format(self.market.target))
            print("*** R/R: {}".format(self.market.reward_risk()))
            print("*** R/R: {}".format(self.market.reward_risk_management()))
            print("*** R/R Large Trades: {}".format(self.market.reward_large_trades()))


        """
        print("----------------------------------------------------------------")
        for i, l in enumerate(self.market.get_state()):
            print("*** Serie *** Frame: {} *** Type: {} ***".format(self.market.info[i]['frame'],
                                                                    self.market.info[i]['type']))
            print(l)
        print("----------------------------------------------------------------")
        """

        """
        if mode == 'system':
            self._render_system()
        elif mode == 'human':
            if self.viewer is None:
                self.viewer = BitcoinTradingGraph(self.bars_array[0])

            self.viewer.render(self.current_bars[0],
                               self.net_worths, self.benchmarks, self.trades)
        """


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    def sample(self):
        return self.action_space.sample()