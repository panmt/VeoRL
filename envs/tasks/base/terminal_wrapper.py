from typing import Dict
from gym import Wrapper
import numpy as np
from abc import ABC, abstractstaticmethod


class TerminalWrapper(Wrapper, ABC):
    def __init__(self, env, max_steps: int = 500, on_death=True, all: Dict = dict(), any: Dict = dict(), stagger_max_steps=False):
        '''
        max_steps: 这是一个整数，表示在没有其他终止条件满足时，环境可以执行的最大步数。一旦达到这个步数，episode 将结束。
        on_death: 这是一个布尔值，指示当检测到“死亡”条件时是否应该终止 episode。
        all_conditions: 这是一个字典，包含了必须全部满足以终止 episode 的条件。如果所有列出的条件都满足，则 episode 结束。
        any_conditions: 这是一个字典，包含了只要满足其中任意一个就应该终止 episode 的条件。如果列出的任何条件满足，则 episode 结束。
        t: 这是一个计数器，用于记录自 episode 开始以来执行的步数。
        curr_max_steps: 这是一个整数，表示当前 episode 的最大步数。它可以通过 stagger_max_steps 参数随机化。
        stagger_max_steps: 这是一个布尔值，指示是否应该随机化 curr_max_steps 的值。如果为真，则 curr_max_steps 将在 [max_steps*3/4, max_steps] 范围内随机选择一个值。
        '''
        super().__init__(env)
        self.wrapper_name = "TerminalWrapper"
        self.max_steps = max_steps
        self.on_death = on_death
        self.all_conditions = all
        self.any_conditions = any
        self.t = 0
        self.curr_max_steps = self.max_steps
        self.stagger_max_steps = stagger_max_steps

    def reset(self):
        print("==========Now is resetting from TerminalWrapper!==========")
        self.t = 0
        if self.stagger_max_steps:
            self.curr_max_steps = np.random.randint((self.max_steps*3)//4, self.max_steps+1)
        else:
            self.curr_max_steps = self.max_steps

        tmp = super().reset()
        print("max_steps:", self.max_steps)
        print("curr_max_steps:", self.curr_max_steps)
        print("on_death(terminate episode when death?):", self.on_death)
        print("all_conditions(when all these conditions are met, terminate episode):", self.all_conditions)
        print("any_conditions(when any of these conditions are met, terminate episode):", self.any_conditions)
        print("==========Resetting from TerminalWrapper is done!==========")
        return tmp

    def step(self, action):
        # print("function step() from terminal_wrapper.py")
        obs, reward, done, info = super().step(action)

        self.t += 1
        done = done or self.t >= self.curr_max_steps
        # done = self.t >= self.curr_max_steps

        if self.on_death:
            done = done or self._check_condition("death", {}, obs)

        if len(self.all_conditions) > 0:
            done = done or all(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.all_conditions.items()
            )

        if len(self.any_conditions) > 0:
            done = done or any(
                self._check_condition(condition_type, condition_info, obs)
                for condition_type, condition_info in self.any_conditions.items()
            )

        return obs, reward, done, info

    def _check_condition(self, condition_type, condition_info, obs):
        if condition_type == "item":
            return self._check_item_condition(condition_info, obs)
        elif condition_type == "blocks":
            return self._check_blocks_condition(condition_info, obs)
        elif condition_type == "death":
            return self._check_death_condition(condition_info, obs)
        else:
            raise NotImplementedError("{} terminal condition not implemented".format(condition_type))

    @abstractstaticmethod
    def _check_item_condition(condition_info, obs):
        raise NotImplementedError()

    @abstractstaticmethod
    def _check_blocks_condition(condition_info, obs):
        raise NotImplementedError()

    @abstractstaticmethod
    def _check_death_condition(condition_info, obs):
        raise NotImplementedError()