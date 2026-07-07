import gymnasium as gym

class TSCEnv(gym.Env):
    def __init__(self, world, agents):
        self.world = world
        self.eng = self.world.eng
        self.agents = agents
        
    def step(self, action):
        self.world.next_step(action)
        
        obs = [agent.get_ob() for agent in self.agents]
        rewards = [agent.get_reward() for agent in self.agents]
        dones = [agent.is_done() for agent in self.agents]
        states = [agent.get_state() for agent in self.agents]
        infos = {'states':states}
        return obs, rewards, dones, infos
 
    def reset(self):
        '''Reset the environment.
        
        Returns:
            observations and states.
        '''
        self.world.reset()
        obs = [agent.get_ob() for agent in self.agents]
        states = [agent.get_state() for agent in self.agents]
        return obs, states

    def stat_through_cnt(self):
        self.world.stat_grid_cnt()
