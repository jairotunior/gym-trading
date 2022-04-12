import gym
import pandas as pd
import numpy as np
from gym_trading.envs.trading_env import TradingEnv
from gym_trading.wrappers import BarColorIndicatorWrapper
from collections import deque, OrderedDict
import settings


class MainPivotIndicatorWrapper(gym.Wrapper):
    name = "Main Pivot Indicator"

    def __init__(self, env: TradingEnv, n_pivots: int = 3):
        gym.Wrapper.__init__(self, env)
        assert hasattr(self.unwrapped, "indicator_list"), "No se ha agregado IndicatorBaseWrapper"
        assert BarColorIndicatorWrapper.name in self.unwrapped.indicator_list, "No se ha agregado el BarColorIndicator"

        self.n_pivots = n_pivots

        # Add the indicator name to list
        self.unwrapped.indicator_list.append(MainPivotIndicatorWrapper.name)

        self.market = self.unwrapped.market

        space = OrderedDict([('upper_pivots', gym.spaces.Box(low=0, high=1, shape=(len(self.market.bars_array),), dtype=np.float)),
                            ('lower_pivots', gym.spaces.Box(low=0, high=1, shape=(len(self.market.bars_array),), dtype=np.float))])

        self.unwrapped.observation_space.spaces = OrderedDict(list(space.items()) + list(self.unwrapped.observation_space.spaces.items()))

        setattr(self.unwrapped, 'pivot_info', [[]]*len(self.market.bars_array))
        setattr(self.unwrapped, 'pivot_case', [0]*len(self.market.bars_array))
        setattr(self.unwrapped, 'selected', [False]*len(self.market.bars_array))

        setattr(self.unwrapped, 'pivot_show', [{
            'up_render_list': [],
            'down_render_list': []
        }] * len(self.market.bars_array))

        self._calculate()

    def _calculate(self):

        selected = self.unwrapped.selected
        pivot_info = self.unwrapped.pivot_info
        pivot_case = self.unwrapped.pivot_case

        pivot_show_cal = [{'down_render_list': [], 'up_render_list': []}] * len(self.market.bars_array)

        for i, df in enumerate(self.market.bars_array):
            color_serie = self.unwrapped.bar_color[i]

            data = {}
            selected[i] = False

            #id_start_date = df[df['dateTime'] == self.market.start_date].index[0]

            for index, (date, row) in enumerate(df.iterrows()):
                if i == 0:
                    continue
                #elif index >= id_start_date:
                #    break

                current_color = color_serie.iloc[index][BarColorIndicatorWrapper.name]
                before_color = color_serie.iloc[index - 1][BarColorIndicatorWrapper.name]

                if selected[i]:
                    if pivot_case[i] == 1:
                        if current_color == 'red':
                            pivot_info[i][-1]['resistence'] = max(pivot_info[i][-1]['resistence'],
                                                                  row['open'])
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'green':
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'doji':
                            selected[i] = False
                    elif pivot_case[i] == 2:
                        if current_color == 'red':
                            pivot_info[i][-1]['resistence'] = max(pivot_info[i][-1]['resistence'],
                                                                  row['open'])
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'green':
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'doji':
                            selected[i] = False
                    elif pivot_case[i] == 3:
                        if current_color == 'red':
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'green':
                            pivot_info[i][-1]['support'] = min(row['open'],
                                                               min(df['open'].iloc[index - 1],
                                                                   df['close'].iloc[index - 1]))
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'doji':
                            selected[i] = False
                    elif pivot_case[i] == 4:
                        if current_color == 'red':
                            pivot_info[i][-1]['resistence'] = max(row['open'],
                                                                  max(df['open'].iloc[index - 1],
                                                                      df['close'].iloc[index - 1]))
                        elif current_color == 'green':
                            selected[i] = False
                            pivot_case[i] = 0
                        elif current_color == 'doji':
                            selected[i] = False

                if not selected[i]:
                    if current_color == 'red':
                        if before_color == 'green':
                            pivot_point = max(df['close'].iloc[index - 1], row['open'])
                            pivot_info[i].append({
                                'dateTime': date, #row['dateTime'],
                                'support': pivot_point,
                                'resistence': pivot_point,
                                'pivot_point': 'resistence'})
                            selected[i] = True
                            pivot_case[i] = 1
                            #if i == 0:
                                #print(pivot_info[i][-1])
                    elif current_color == 'green':
                        if before_color == 'red':
                            pivot_point = min(df['close'].iloc[index - 1], row['open'])
                            pivot_info[i].append({
                                'dateTime': date, #row['dateTime'],
                                'support': pivot_point,
                                'resistence': pivot_point,
                                'pivot_point': 'support'})
                            selected[i] = True
                            pivot_case[i] = 2
                            #if i == 0:
                                #print(pivot_info[i][-1])
                    elif current_color == 'doji':
                        if before_color == 'red':
                            pivot_point = min(df['close'].iloc[index - 1], row['open'])
                            pivot_info[i].append({
                                'dateTime': date, #row['dateTime'],
                                'support': pivot_point,
                                'resistence': pivot_point,
                                'pivot_point': 'support'})
                            selected[i] = True
                            pivot_case[i] = 3
                            #if i == 0:
                                #print(pivot_info[i][-1])
                        elif before_color == 'green':
                            pivot_point = max(df['close'].iloc[index - 1], row['open'])
                            pivot_info[i].append({
                                'dateTime': date, #row['dateTime'],
                                'support': pivot_point,
                                'resistence': pivot_point,
                                'pivot_point': 'resistence'})
                            selected[i] = True
                            pivot_case[i] = 4
                            #if i == 0:
                                #print(pivot_info[i][-1])

            num_up_pivots = [0] * len(self.market.bars_array)
            num_down_pivots = [0] * len(self.market.bars_array)

            for i, df in enumerate(self.market.bars_array):
                up_price = self.market.get_close(i, 0)
                down_price = self.market.get_close(i, 0)

                if len(pivot_info[i]) > 0:
                    for p in reversed(pivot_info[i]):
                        if up_price < p['resistence'] and num_up_pivots[i] < self.n_pivots:
                            pivot_show_cal[i]['up_render_list'].append(p)
                            up_price = p['resistence'] + settings.SYMBOL_LIST[self.market.symbol]['tickSize']
                            num_up_pivots[i] += 1
                            continue

                        if down_price > p['support'] and num_down_pivots[i] < self.n_pivots:
                            pivot_show_cal[i]['down_render_list'].append(p)
                            down_price = p['support'] - settings.SYMBOL_LIST[self.market.symbol]['tickSize']
                            num_down_pivots[i] += 1
                            continue

        self.unwrapped.pivot_show = pivot_show_cal

        #print(self.unwrapped.pivot_show)

    def step(self, action):
        #print("MainPivot Start")
        selected = self.unwrapped.selected
        pivot_info = self.unwrapped.pivot_info
        pivot_case = self.unwrapped.pivot_case
        pivot_show = self.unwrapped.pivot_show

        for i, df in enumerate(self.market.bars_array):
            current_color = self.unwrapped.get_color_bar(i, 0)
            before_color = self.unwrapped.get_color_bar(i, 1)

            if selected[i]:
                if pivot_case[i] == 1:
                    if current_color == 'red':
                        pivot_info[i][-1]['resistence'] = max(pivot_info[i][-1]['resistence'], self.market.get_open(i, 0))
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'green':
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'doji':
                        selected[i] = False
                elif pivot_case[i] == 2:
                    if current_color == 'red':
                        pivot_info[i][-1]['resistence'] = max(pivot_info[i][-1]['resistence'],
                                                                   self.market.get_open(i, 0))
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'green':
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'doji':
                        selected[i] = False
                elif pivot_case[i] == 3:
                    if current_color == 'red':
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'green':
                        pivot_info[i][-1]['support'] = min(self.market.get_open(i, 0),
                                                           min(self.market.get_open(i, 1), self.market.get_close(i, 1)))
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'doji':
                        selected[i] = False
                elif pivot_case[i] == 4:
                    if current_color == 'red':
                        pivot_info[i][-1]['resistence'] = max(self.market.get_open(i, 0),
                                                              max(self.market.get_open(i, 1), self.market.get_close(i, 1)))
                    elif current_color == 'green':
                        selected[i] = False
                        pivot_case[i] = 0
                    elif current_color == 'doji':
                        selected[i] = False

            if not selected[i]:
                if current_color == 'red':
                    if before_color == 'green':
                        pivot_point = max(self.market.get_close(i, 1), self.market.get_open(i, 0))
                        pivot_info[i].append({
                            'dateTime': self.market.get_date(i, 0),
                            'support': pivot_point,
                            'resistence': pivot_point,
                            'pivot_point': 'resistence'})
                        selected[i] = True
                        pivot_case[i] = 1
                        #if i == 0:
                            #print(pivot_info[i][-1])
                elif current_color == 'green':
                    if before_color == 'red':
                        pivot_point = min(self.market.get_close(i, 1), self.market.get_open(i, 0))
                        pivot_info[i].append({
                            'dateTime': self.market.get_date(i, 0),
                            'support': pivot_point,
                            'resistence': pivot_point,
                            'pivot_point': 'support'})
                        selected[i] = True
                        pivot_case[i] = 2
                        #if i == 0:
                            #print(pivot_info[i][-1])
                elif current_color == 'doji':
                    if before_color == 'red':
                        pivot_point = min(self.market.get_close(i, 1), self.market.get_open(i, 0))
                        pivot_info[i].append({
                            'dateTime': self.market.get_date(i, 0),
                            'support': pivot_point,
                            'resistence': pivot_point,
                            'pivot_point': 'support'})
                        selected[i] = True
                        pivot_case[i] = 3
                        #if i == 0:
                            #print(pivot_info[i][-1])
                    elif before_color == 'green':
                        pivot_point = max(self.market.get_close(i, 1), self.market.get_open(i, 0))
                        pivot_info[i].append({
                            'dateTime': self.market.get_date(i, 0),
                            'support': pivot_point,
                            'resistence': pivot_point,
                            'pivot_point': 'resistence'})
                        selected[i] = True
                        pivot_case[i] = 4
                        #if i == 0:
                            #print(pivot_info[i][-1])

        pivot_show_cal = [{ 'down_render_list': [], 'up_render_list': []}] * len(self.market.bars_array)

        num_up_pivots = [0] * len(self.market.bars_array)
        num_down_pivots = [0] * len(self.market.bars_array)

        for i in range(len(self.market.bars_array)):
            up_price = self.market.get_close(i, 0)
            down_price = self.market.get_close(i, 0)

            if len(pivot_info[i]) > 0:
                for p in reversed(pivot_info[i]):
                    if up_price < p['resistence'] and num_up_pivots[i] < self.n_pivots:
                        pivot_show_cal[i]['up_render_list'].append(p)
                        up_price = p['resistence'] + settings.SYMBOL_LIST[self.market.symbol]['tickSize']
                        num_up_pivots[i] += 1
                        continue

                    if down_price > p['support'] and num_down_pivots[i] < self.n_pivots:
                        pivot_show_cal[i]['down_render_list'].append(p)
                        down_price = p['support'] - settings.SYMBOL_LIST[self.market.symbol]['tickSize']
                        num_down_pivots[i] += 1
                        continue

        pivot_show = pivot_show_cal

        up_pivot = np.zeros(shape=(len(self.market.bars_array), ))
        down_pivot = np.zeros(shape=(len(self.market.bars_array), ))

        for i, series in enumerate(pivot_show_cal):
            for down in series['down_render_list']:
                down_pivot[i] = down['support']

            for up in series['up_render_list']:
                up_pivot[i] = up['resistence']

        state, reward, done, _ = self.env.step(action)
        state['upper_pivots'] = up_pivot
        state['lower_pivots'] = down_pivot

        return state, reward, done, _

    def reset(self, **kwargs):
        #print("MainPivot Start")

        self._calculate()

        up_pivot = np.zeros(shape=(len(self.market.bars_array), ))
        down_pivot = np.zeros(shape=(len(self.market.bars_array), ))

        for i, series in enumerate(self.unwrapped.pivot_show):
            for down in series['down_render_list']:
                down_pivot[i] = down['support']

            for up in series['up_render_list']:
                up_pivot[i] = up['resistence']

        state = self.env.reset()

        state['upper_pivots'] = up_pivot
        state['lower_pivots'] = down_pivot

        #print("MainPivot End")
        return state