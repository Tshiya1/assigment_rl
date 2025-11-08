import gym

class CrafterRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.health = None
        self.food = None
        self.drink = None
        self.energy = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.health = 9
        self.food = 9
        self.drink = 9
        self.energy = 9
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Encourage maintatining it's self
        shaped_reward = reward
        if self.health < info["inventory"]["health"]:
          shape_reward = shape_reward + 0.5
        elif info["inventory"]["health"] == 9:
          shape_reward = shape_reward + 0.5

        if self.food < info["inventory"]["food"]:
          shape_reward = shape_reward + 0.5
        elif info["inventory"]["food"] == 9:
          shape_reward = shape_reward + 0.5

        if self.drink < info["inventory"]["drink"]:
          shape_reward = shape_reward + 0.5
        elif info["inventory"]["drink"] == 9:
          shape_reward = shape_reward + 0.5

        if self.energy < info["inventory"]["energy"]:
          shape_reward = shape_reward + 0.5
        elif info["inventory"]["energy"] == 9:
          shape_reward = shape_reward + 0.5

        self.health = info["inventory"]["health"]
        self.food = info["inventory"]["food"]
        self.drink = info["inventory"]["drink"]
        self.energy = info["inventory"]["energy"]
        
        return obs, shaped_reward, done, info