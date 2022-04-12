from gym.envs.registration import register
import settings

# Env registration
# ==========================

register(
    id='TradingEnv-v0',
    entry_point='gym_trading.envs:TradingEnv',
)

register(
    id='TradingEnv-v1',
    entry_point='gym_trading.envs:TradingEnv',
    kwargs = { 'use_settings': False }
)
