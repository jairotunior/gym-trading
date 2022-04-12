import gym
import pandas as pd
import numpy as np
from types import MethodType
from collections import deque

def get_color_bar(self, bar_index=0, index=0):
    return self.bar_color[bar_index].iloc[self.market.current_bars[bar_index] - index - 1, :][BarColorIndicatorWrapper.name]

class BarColorIndicatorWrapper(gym.Wrapper):
    name = "Bar Color Indicator"

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert hasattr(self.unwrapped, "indicator_list"), "No se ha agregado IndicatorBaseWrapper"

        # Agrega una serie en el environment
        setattr(self.unwrapped, 'bar_color', deque(maxlen=100))
        setattr(self.unwrapped, 'get_color_bar', MethodType(get_color_bar, self.unwrapped))

        # Add the indicator name to list
        self.unwrapped.indicator_list.append(BarColorIndicatorWrapper.name)

        # Process the pivots
        self._calculate()

    def _calculate(self):
        for i, info in enumerate(self.unwrapped.market.info):

            df = self.unwrapped.market.bars_array[i]
            conditions = [
                (df['close'] > df['open']),
                (df['close'] < df['open'])
            ]
            choices = ['green', 'red']
            #self.unwrapped.market.bars_array[i][self.name] = np.select(conditions, choices, default='doji')

            data = {
                'dateTime': self.unwrapped.market.bars_array[i].index,
                BarColorIndicatorWrapper.name: np.select(conditions, choices, default='doji')
            }

            colors = pd.DataFrame(data, columns=['dateTime', BarColorIndicatorWrapper.name])
            colors = colors.set_index('dateTime')
            self.unwrapped.bar_color.append(colors)

    def observation(self, observation):
        return observation

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        #print("BarColor Start")
        #print("BarColor End")
        return self.env.reset()