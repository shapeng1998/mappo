import numpy as np


class Env(object):
    """
    # 环境中的智能体
    """

    def __init__(self, i):
        self.agent_num = 2  # 设置智能体(小飞机)的个数，这里设置为两个
        self.obs_dim = 14  # 设置智能体的观测纬度
        self.action_dim = 5  # 设置智能体的动作纬度，这里假定为一个五个纬度的

    def reset(self):
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs = np.random.random(size=(14,))
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs

    def step(self, actions):
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = []
        sub_agent_info = []
        for i in range(self.agent_num):
            sub_agent_obs.append(np.random.random(size=(14,)))
            sub_agent_reward.append([np.random.rand()])
            sub_agent_done.append(False)
            sub_agent_info.append({})
        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
