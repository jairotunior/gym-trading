import numpy as np
import pandas as pd
import math
import gym
import gym.spaces

import settings

import joblib

from sklearn.preprocessing import StandardScaler
#from empyrical import sortino_ratio, calmar_ratio, omega_ratio

from gym_trading.envs.trading_env import MarketPosition

from collections import deque


class RewardScaler(gym.RewardWrapper):

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        #reward_risk = self.reward_risk_management()
        #reward_large_trade = self.reward_large_trades()
        return reward * 0.00001
"""
class RewardRiskRatios(gym.RewardWrapper):

    def __init__(self, env, reward_fc='sortino'):
        gym.RewardWrapper.__init__(self, env)
        self.reward_fc = reward_fc

    def reward(self, reward):
        length = self.env.market.current_bars[0]

        # Get the account balance per candle
        returns = np.diff(self.env.market.net_worths)

        if np.count_nonzero(returns) < 1:
            return 0

        if self.reward_fc == 'sortino':
            reward = sortino_ratio(returns, annualization=250*24)
        elif self.reward_fc == 'calmar':
            reward = calmar_ratio(returns, annualization=250*24)
        elif self.reward_fc == 'omega':
            reward = omega_ratio(returns, annualization=250*24)
        else:
            reward = returns[-1]

        result = reward if np.isfinite(reward) else 0

        return result
"""

class RewardMarketRatios(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.env = env

    def reward(self, reward):
        risk = 0
        profit = 0
        _reward = 0

        print("********************************************************")

        if self.env.market.market_position != MarketPosition.Flat:
            risk = self.env.market.price_entry - self.env.market.stoploss + 1
            profit = self.env.market.current_price - self.env.market.price_entry

            # No hay riesgo. El stoploss esta por encima del breakeven
            if risk < 0:
                if profit > 0:
                    print("*1Numerator: {} - Denominator: {}".format(profit + self.env.market.risk_init,
                                                                    profit + self.env.market.risk_init + risk))
                    _reward = ((profit + self.env.market.risk_init) / (profit + self.env.market.risk_init + risk)) * 5
                else:
                    print("*2Numerator: {} - Denominator: {}".format(self.env.market.risk_init,
                                                                    self.env.market.risk_init + risk))
                    _reward = (self.env.market.risk_init) / (self.env.market.risk_init + risk)
                    if _reward == float('inf'):
                        _reward = 0
            # El stoploss esta por debajo del breakeven
            elif risk > 0:
                print("**Numerator: {} - Denominator: {}".format(profit + self.env.market.risk_init, risk))
                _reward = (profit + self.env.market.risk_init) / risk
            # El stoploss esta en Breakeven
            else:
                print("***Numerator: {} - Denominator: {}".format(0, 0))
                _reward = 20

        if self.env.market.space[self.env.last_action] == "close":
            print("****CLOSE POSITION")
            _reward = self.env.market.account_balance / settings.ACCOUNT_MONEY
        """
        elif self.env.market.market_position == MarketPosition.Flat:
            if self.env.market.stoploss_position:
                _reward = -1
        """

        if self.env.market._dones_maximum_loss():
            print("*****MAXIMUM LOSS")
            _reward = -100

        print("Profit: {} -  Risk Init: {} -  Risk: {} - Reward: {}".format(
            profit, self.env.market.risk_init, risk, _reward))

        self.env.current_reward = _reward

        return _reward



class PreprocessObservation(gym.ObservationWrapper):

    def __init__(self, env, scalers=None):
        gym.ObservationWrapper.__init__(self, env)

        # Columns of return observation
        self.columns = ['open', 'high', 'low', 'close', 'volume']
        self.columns_preprocesses = ['candle_size', 'gap_size', 'high_size', 'low_size', 'volume_size']

        self.scalers = None

        if scalers is not None:
            self.scalers = scalers


    """
        Observation variable is type pandas.core.frame.DataFrame
    """
    def observation(self, observation):
        #print("Preprocess Start")
        new_obs = []

        #print(observation['series'])

        for i, obs in enumerate(observation['series']):
            # Format of representation
            # Size Candle = Math.abs(Open(t) - Close(t))
            # Gap Size = Close(t - 1) - Open(t)
            # High Size = High(t) - max(Open(t), Close(t))
            # Low Size = Low(t) - min(Open(t), Close(t))

            ob = obs[self.columns]
            ob.iloc[:, :]['candle_size'] = obs['close'] - obs['open']
            ob.iloc[:, :]['gap_size'] = obs['open'] - obs['close'].shift(1)
            ob.iloc[:, :]['high_size'] = obs['high'] - obs[['open', 'close']].max(axis=1)
            ob.iloc[:, :]['low_size'] = obs[['open', 'close']].min(axis=1) - obs['low']
            ob.iloc[:, :]['volume_size'] = obs['volume'].values
            ob.volume_size = ob.volume_size.astype('float64')

            ob = ob.fillna(0)

            if self.scalers is not None:
                if any(self.scalers[i]):
                    for column in self.columns_preprocesses:
                        #print(column)
                        ob_values = ob[[column]].values
                        ob[column] = self.scalers[i][column].inverse_transform(ob_values)

            #new_obs.append(np.reshape(obs[self.columns].values, (*obs[self.columns].shape, 1)))

            if self.unwrapped.transposed:
                ob = ob[self.columns_preprocesses].transpose()
            else:
                ob = ob[self.columns_preprocesses]

            obs_array = np.reshape(ob.values, (*ob.shape, 1))

            new_obs.append(obs_array)

        observation['series'] = new_obs

        return observation


class ActionDiscretizer(gym.ActionWrapper):

    def __init__(self, env):
        super(ActionDiscretizer, self).__init__(env)

        space = ["BUY", "SELL", "NOTHING", "CLOSE", "TRAILING_UP", "TRAILING_DOWN"]
        actions = [['BUY'], ['SELL'], ['NOTHING'], ['CLOSE'], ["TRAILING_UP"], ["TRAILING_DOWN"]]
        self._actions = []

        for action in actions:
            arr = np.array([False] * len(space))
            for s in space:
                arr[space.index(s)] = True
            self._actions.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        return self._actions[action].copy()


class StackWrapper(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.observation_space = env.observation_space
        self.shp = self.observation_space.spaces['series'][0].shape
        self.observation_space.spaces['series'] = gym.spaces.Tuple(
            tuple([gym.spaces.Box(low=0, high=1, shape=(*self.shp[:-1], k), dtype=np.float)] + [ser for i, ser in enumerate(self.observation_space.spaces['series']) if i != 0])
        )

        #print("Nueva Shape")
        #print(self.observation_space.spaces['series'][0].shape)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob['series'][0])
        return self._get_ob(ob)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob['series'][0])
        return self._get_ob(ob), reward, done, info

    def _get_ob(self, ob):
        obs = np.zeros(self.shp, dtype=np.float)

        for i, frame in enumerate(self.frames):
            obs[:][:] = frame

        ob['series'][0] = obs

        return ob