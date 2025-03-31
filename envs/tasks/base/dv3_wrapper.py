from gym import Wrapper
import gym.spaces as spaces
import numpy as np
from abc import ABC, abstractmethod
import threading
import cv2
import copy
from collections import OrderedDict

BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.])),
    "turn_down": dict(camera=np.array([10.0, 0.])),
    "turn_left": dict(camera=np.array([0., -10.0])),
    "turn_right": dict(camera=np.array([0., 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    # "place_dirt": dict(place="dirt"),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    'camera': np.array([0., 0.]), 
    'smelt': 'none', 
    'craft': 'none', 
    'craft_with_table': 'none', 
    'forward': np.array(0), 
    'back': np.array(0), 
    'left': np.array(0), 
    'right': np.array(0), 
    'jump': np.array(0), 
    'sneak': np.array(0), 
    'sprint': np.array(0), 
    'use': np.array(0), 
    'attack': np.array(0), 
    'drop': 0, 
    'swap_slot': OrderedDict([('source_slot', 0), ('target_slot', 0)]), 
    'pickItem': 0, 
    'hotbar.1': 0, 
    'hotbar.2': 0, 
    'hotbar.3': 0, 
    'hotbar.4': 0, 
    'hotbar.5': 0, 
    'hotbar.6': 0, 
    'hotbar.7': 0, 
    'hotbar.8': 0, 
    'hotbar.9': 0,
}


class DV3Wrapper(Wrapper, ABC):
    def __init__(self, env, repeat=1, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70)):
        super().__init__(env)
        self.wrapper_name = "DV3Wrapper"

        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        # print("actions:", actions)
        # actions 中包含了 BASIC_ACTIONS 中提到的动作，键为动作名，值为动作对应的操作（NOOP_ACTION为一个误动作操作的例子）
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())

        # print("self._action_names:", self._action_names)
        # print("self._action_values:", self._action_values)

        self.observation_space = spaces.Dict(
            {
                'image': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                # 'is_first': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                # 'is_last': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                # 'is_terminal': spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                # 'concentration_score': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                # 'zoom_in_prob': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                # 'num_above_threshold': spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True

        # self._noop_action = 
        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        # with threading.Lock():
        obs = self.env.reset()
        # obs["is_first"] = True
        # obs["is_last"] = False
        # obs["is_terminal"] = False
        obs = self._obs(obs)

        # print("reset_from_DV3Wrapper:", self._action_values)
        # assert self._action_values == self._action_values_check

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def step(self, action):
        # print("action:", action)
        # action = action.copy()
        # action_jump = [action_value["jump"] for action_value in self._action_values]
        # print("action_jump:", action_jump)
        # assert self._action_values[0]["jump"] == 0
        # action = self._action_values[action]
        action = copy.deepcopy(self._action_values[action])
        # print("------", self._action_values)
        # print("action_00:", action["jump"])

        # assert 1==2
        # assert self._action_values == self._action_values_check

        action = self._action(action)
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            # print("following!", act)
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        # obs["is_first"] = False
        # obs["is_last"] = bool(done)
        # obs["is_terminal"] = bool(info.get("is_terminal", done))

        obs = self._obs(obs)
        assert "pov" not in obs, list(obs.keys())
        return obs, reward, done, info


    
    def _obs(self, obs):
        image = obs['rgb'] # 3 * H * W
        image = image.transpose(1, 2, 0).astype(np.uint8) # H * W * 3
        image = cv2.resize(image, (64, 64)) # 64 * 64 * 3

        obs = {
            'image': image,
            # 'is_first': obs['is_first'],
            # 'is_last': obs['is_last'],
            # 'is_terminal': obs['is_terminal'],
            # 'concentration_score': obs['concentration_score'],
            # 'zoom_in_prob': obs['zoom_in_prob'],
            # 'num_above_threshold': obs['num_above_threshold'],
        }

        for key, value in obs.items():
            space = self.observation_space[key]
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            assert (key, value, value.dtype, value.shape, space)
        return obs

    def _action(self, action):
        # print("action_01:", action["jump"])
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"] and self._sticky_jump_counter == 0:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                # print("sticky_jump_counter:", self._sticky_jump_counter)
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        # print("action_02:", action["jump"])
        # print(action["camera"][0])
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        # print("self.pitch:", self._pitch)

        # print(f'self.pitch: {self._pitch}, action["camera"]: {action["camera"]}')

        return action


    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                if key not in action:
                    action[key] = default
        return actions

        