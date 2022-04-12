import gym

class IndicatorBaseWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        # Agrega una variable que es necesaria para utilizar indicadores
        setattr(self.unwrapped, 'indicator_list', [])

    def reset(self, **kwargs):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)