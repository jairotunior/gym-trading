import gym
import numpy as np
from types import MethodType
from gym_trading.envs.trading_env import MarketPosition

#region ********************************** Backtesting Metrics **********************************

def winning_trades(self, market_position = None):
    if market_position is None:
        mask = (self.trades['profit'] > 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
    return self.trades[mask]['profit'].count()

def losing_trades(self, market_position = None):
    if market_position is None:
        mask = (self.trades['profit'] < 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
    return self.trades[mask]['profit'].count()

def even_trades(self, market_position = None):
    if market_position is None:
        mask = (self.trades['profit'] == 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] == 0)
    return self.trades[mask]['profit'].count()

def net_profit(self, market_position = None):
    if market_position is None:
        return self.trades['profit'].sum()
    else:
        mask = (self.trades['market_position'] == market_position)

    return self.trades[mask]['profit'].sum()

def gross_profit(self, market_position = None):
    if market_position is None:
        mask = (self.trades['gross_profit'] > 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['gross_profit'] > 0)
    return self.trades[mask]['gross_profit'].sum()

def gross_loss(self, market_position = None):
    if market_position is None:
        mask = (self.trades['gross_profit'] < 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['gross_profit'] < 0)
    return self.trades[mask]['gross_profit'].sum()

def commisions(self, market_position = None):
    if market_position is None:
        return self.trades['commisions'].sum()

    mask = (self.trades['market_position'] == market_position)
    return self.trades[mask]['commisions'].sum()

def average_winning_trades(self, market_position = None):
    if market_position is None:
        mask = (self.trades['profit'] > 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
    return self.trades[mask]['profit'].mean()

def average_losing_trades(self, market_position = None):
    if market_position is None:
        mask = (self.trades['profit'] < 0)
    else:
        mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
    return self.trades[mask]['profit'].mean()

def ratio_win_loss(self, market_position = None):
    return np.abs(self.average_winning_trades(market_position) / self.average_losing_trades(market_position))

def max_consecutive_winners(self, market_position = None):
    self.trades['consecutive'] = np.sign(self.trades['profit'])
    self.trades['count_consecutive'] = self.trades['consecutive'] * (self.trades['consecutive'].groupby((self.trades['consecutive'] != self.trades['consecutive'].shift()).cumsum()).cumcount() + 1)
    mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] > 0)
    return self.trades[mask]['count_consecutive'].max()

def max_consecutive_loser(self, market_position = None):
    self.trades['consecutive'] = np.sign(self.trades['profit'])
    self.trades['count_consecutive'] = self.trades['consecutive'] * (self.trades['consecutive'].groupby((self.trades['consecutive'] != self.trades['consecutive'].shift()).cumsum()).cumcount() + 1)

    mask = (self.trades['market_position'] == market_position) & (self.trades['profit'] < 0)
    return self.trades[mask]['count_consecutive'].max()

#endregion


class MetricsWrapper(gym.Wrapper):

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.market = self.unwrapped.market

        setattr(self.market, "winning_trades", MethodType(winning_trades, self.market))
        setattr(self.market, "losing_trades", MethodType(losing_trades, self.market))
        setattr(self.market, "even_trades", MethodType(even_trades, self.market))
        setattr(self.market, "net_profit", MethodType(net_profit, self.market))
        setattr(self.market, "gross_profit", MethodType(gross_profit, self.market))
        setattr(self.market, "gross_loss", MethodType(gross_loss, self.market))
        setattr(self.market, "commisions", MethodType(commisions, self.market))
        setattr(self.market, "average_winning_trades", MethodType(average_winning_trades, self.market))
        setattr(self.market, "average_losing_trades", MethodType(average_losing_trades, self.market))
        setattr(self.market, "ratio_win_loss", MethodType(ratio_win_loss, self.market))
        setattr(self.market, "max_consecutive_winners", MethodType(max_consecutive_winners, self.market))
        setattr(self.market, "max_consecutive_loser", MethodType(max_consecutive_loser, self.market))

    def render(self, mode='human', **kwargs):

        print("{:*^80}".format(" SUMMARY "))
        print("*** Datetime: {}".format(self.market.current_date))
        print("*** Market Position: {}".format(self.market.market_position))
        print("*** Last Price: {}".format(self.market.current_price))
        print("*** Action: {}".format(self.unwrapped.last_action))
        print("*** Account Balance: {}".format(self.market.account_balance))
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("*** {:<20} {:>12} {:>12} {:>12}".format("PERFORMANCE", "TOTAL", "LONG", "SHORT"))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Total Net Profit",
                                                       self.market.net_profit(),
                                                       self.market.net_profit(MarketPosition.Long),
                                                       self.market.net_profit(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Gross Profit",
                                                       self.market.gross_profit(),
                                                       self.market.gross_profit(MarketPosition.Long),
                                                       self.market.gross_profit(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Gross Loss",
                                                       self.market.gross_loss(),
                                                       self.market.gross_loss(MarketPosition.Long),
                                                       self.market.gross_loss(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Commisions",
                                                       self.market.commisions(),
                                                       self.market.commisions(MarketPosition.Long),
                                                       self.market.commisions(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Winning Trades",
                                                       self.market.winning_trades(),
                                                       self.market.winning_trades(MarketPosition.Long),
                                                       self.market.winning_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Losing Trades",
                                                       self.market.losing_trades(),
                                                       self.market.losing_trades(MarketPosition.Long),
                                                       self.market.losing_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12} {:>12} {:>12}".format("Even Trades",
                                                       self.market.even_trades(),
                                                       self.market.even_trades(MarketPosition.Long),
                                                       self.market.even_trades(MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Avg. Winning Trades",
                                                                self.market.average_winning_trades(),
                                                                self.market.average_winning_trades(MarketPosition.Long),
                                                                self.market.average_winning_trades(
                                                                    MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Avg. Losing Trades",
                                                                self.market.average_losing_trades(),
                                                                self.market.average_losing_trades(MarketPosition.Long),
                                                                self.market.average_losing_trades(
                                                                    MarketPosition.Short)))
        print("*** {:<20} {:>12.2f} {:>12.2f} {:>12.2f}".format("Ratio Win/Loss",
                                                                self.market.ratio_win_loss(),
                                                                self.market.ratio_win_loss(MarketPosition.Long),
                                                                self.market.ratio_win_loss(MarketPosition.Short)))

        if self.market.market_position != MarketPosition.Flat:
            print("--------------------------TRADES--------------------------------")
            print("*** Datetime: {}".format(self.market.current_trade['date_entry']))
            print("*** Contracts: {}".format(self.market.current_trade['contracts']))
            print("*** Symbol: {}".format(self.market.current_trade['symbol']))
            print("*** Price Entry: {}".format(self.market.current_trade['price_entry']))
            print("*** Price Exit: {}".format(self.market.current_trade['price_exit']))
            print("*** Stoploss: {}".format(self.market.stoploss))

        """
        print("----------------------------------------------------------------")
        for i, l in enumerate(self.market.get_state()):
            print("*** Serie *** Frame: {} *** Type: {} ***".format(self.market.info[i]['frame'],
                                                                    self.market.info[i]['type']))
            print(l)
        print("----------------------------------------------------------------")
        """


    def step(self, action):
        return self.unwrapped.step(action)

    def reset(self):
        return self.unwrapped.reset()