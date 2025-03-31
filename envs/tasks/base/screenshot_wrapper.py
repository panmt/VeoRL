from typing import Dict
from gym import Wrapper
from abc import ABC, abstractstaticmethod
from datetime import datetime
from PIL import Image
import numpy as np
import os

class ScreenshotWrapper(Wrapper, ABC):
    def __init__(self, env, log_dir, reset_flag=False, step_flag=False, save_freq=1,**kwargs):
        print("==========checkpoint01311==========")
        super().__init__(env)
        self.wrapper_name = "ScreenshotWrapper"
        self.reset_flag = reset_flag
        self.step_flag = step_flag
        self.ff = False
        self.save_freq = save_freq
        self.freq = 0
        print("==========checkpoint01312==========")
        # self.log_dir = os.path.join(log_dir, "screenshot")
        self.log_dir = "screenshot"
        print("==========checkpoint01313==========")

    def reset(self):
        print("==========Now is resetting from ScreenshotWrapper!==========")

        obs = super().reset()
        
        # if self.reset_flag:
        #     self.screenshot(obs, "reset")

        self.ff = True
        self.freq = 0

        print("==========Resetting from ScreenshotWrapper is done!==========")
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.freq += 1

        if self.reset_flag and self.ff:
            self.screenshot(obs, "reset")
            self.ff = False

        if self.step_flag and self.freq % self.save_freq == 0:
            self.screenshot(obs, "step")

        return obs, reward, done, info
    


    def screenshot(self, obs, type):
        # 确保 log_dir 目录存在
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # 获取当前帧
        img = self._get_curr_frame(obs)
        # 将 numpy 数组转换为 PIL 图像
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        # 生成文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 截取前三位微秒，转换为毫秒
        filename = f"{type}_{current_time}.png"
        filepath = os.path.join(self.log_dir, filename)
        
        # 保存图像
        img.save(filepath)
        
        # print(f"Screenshot saved at {filepath}")
    
    @abstractstaticmethod
    def get_resolution():
        raise NotImplementedError()

    @abstractstaticmethod
    def _get_curr_frame(obs):
        raise NotImplementedError()