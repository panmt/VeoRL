from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod


class RewardWrapper(Wrapper, ABC):
    def __init__(self, env, item_rewards: Dict[str, Dict[str, int]] = dict()):
        super().__init__(env)
        self.wrapper_name = "RewardWrapper"
        self.item_rewards = item_rewards
        self.last_inventory = None
        self.reward_count = None # 记录了自环境最后一次重置以来，玩家因为特定物品的数量变化而获得的奖励次数

    def reset(self):
        print("==========Now is resetting from RewardWrapper!==========")
        self.last_inventory = {item: 0 for item in self.item_rewards}
        self.reward_count = {item: 0 for item in self.item_rewards}

        tmp = super().reset()
        print("item_rewards:", self.item_rewards)
        print("==========Resetting from RewardWrapper is done!==========")
        return tmp

    def step(self, action):
        # print("action_from_reward_wrapper:", action)
        # print("function step() from reward_wrapper.py")
        obs, reward, done, info = super().step(action)

        for item in self.item_rewards:
            curr_inv = self._get_item_count(obs, item)
            item_diff = curr_inv - self.last_inventory[item]
            if "quantity" in self.item_rewards[item]:
                item_diff = min(item_diff, self.item_rewards[item]["quantity"] - self.reward_count[item])
            item_reward = self.item_rewards[item]["reward"] if "reward" in self.item_rewards[item] else 1
            reward += item_diff * item_reward
            self.reward_count[item] += item_diff
            self.last_inventory[item] = curr_inv

        return obs, reward, done, info

    @abstractstaticmethod
    def _get_item_count(obs, item):
        raise NotImplementedError()
