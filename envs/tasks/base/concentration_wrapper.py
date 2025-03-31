from gym import Wrapper
import torch as th
import numpy as np

class ConcentrationWrapper(Wrapper):
    def __init__(self, env, concentration, prompts=None, gaussian=True, fusion=False, zoom_in=True, dense_reward=.01, clip_target=23, clip_min=21, smoothing=50, **kwargs):
        super().__init__(env)
        self.concentration = concentration # ConcentrationReward
        self.wrapper_name = "ConcentrationWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        self.gaussian = gaussian
        self.fusion = fusion
        self.zoom_in = zoom_in

        self.clip_target = clip_target
        self.clip_min = clip_min

        self.buffer = None
        self.zoom_in_buffer = None
        self.last_score = 0
        self.last_zoom_in_score = 0

    def reset(self, **kwargs):
        print("==========Now is resetting from ConcentrationWrapper!==========")

        self.buffer = None
        self.zoom_in_buffer = None
        self.last_score = 0
        self.last_zoom_in_score = 0

        print("self.prompt", self.prompt)
        obs = self.env.reset(**kwargs)

        score, zoom_in_prob, num_above_threshold = self.concentration.get_reward(obs, self.prompt)
            # print("score: ", score)

        obs['zoom_in_prob'] = zoom_in_prob
        obs['num_above_threshold'] = num_above_threshold   

        self.buffer = self._insert_buffer(self.buffer, score)
        score = self._get_score()

        if score > self.last_score: 
            obs['concentration_score'] = score * self.dense_reward
            self.last_score = score
        else:
            obs['concentration_score'] = 0

        # print("env.observation_space: ", self.env.observation_space)
        print("==========Resetting from ConcentrationWrapper is done!==========")
        
        if self.fusion:
            
            if self.concentration.mask is not None:
                mask = self.concentration.mask
            else:
                mask = np.zeros((160, 256)) # 160*256

            # 假设obs["rgb"]和mask已经被定义
            rgb = obs["rgb"]  # (3, 160, 256) numpy array
            mask = mask  # (160, 256) numpy array

            # 将mask增加一个维度以匹配rgb的维度，变为(1, 160, 256)
            mask_expanded = np.expand_dims(mask, axis=0)

            # 检查mask是否需要转换类型，与rgb的数据类型保持一致
            if mask_expanded.dtype != rgb.dtype:
                mask_expanded = mask_expanded.astype(rgb.dtype)

            # 沿着通道维度堆叠rgb数组和扩展后的mask
            obs["rgb"] = np.concatenate((rgb, mask_expanded), axis=0)

        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            score, zoom_in_prob, num_above_threshold = self.concentration.get_reward(obs, self.prompt)
            # print("score: ", score)
            zoomed_image = self.concentration.generate_zoom_in_frame(obs)
            
            obs['zoom_in_prob'] = zoom_in_prob
            obs['num_above_threshold'] = num_above_threshold   

            self.buffer = self._insert_buffer(self.buffer, score)
            score = self._get_score()

            if score > self.last_score: 
                self.last_score = score
                obs['concentration_score'] = self.dense_reward * score
                if self.gaussian:
                    reward += self.dense_reward * score
            else:
                obs['concentration_score'] = 0
                        

            if self.zoom_in:
                # print("zoom_in")
                # 加入对于价值图最高点 zoom in 模拟探索的 Clip 奖励
                self.concentration.frames_process(obs)
                self.concentration.compute_zoom_in_score(self.prompt)
                # print (self.concentration.zoom_in_score) # tensor
                zoom_in_score_tmp = 0           

                # 检查是否为标量张量 (没有维度)
                if self.concentration.zoom_in_score.dim() == 0:
                    # 由于是标量，直接处理这个标量张量
                    zoom_in_score_tmp += (max(self.concentration.zoom_in_score.item() - self.clip_min, 0) / (self.clip_target - self.clip_min))
                else:
                    # 处理一维张量
                    for i in range(self.concentration.zoom_in_score.size(0)):
                        zoom_in_score_tmp += (max(self.concentration.zoom_in_score[i].item() - self.clip_min, 0) / (self.clip_target - self.clip_min))
                
                self.zoom_in_buffer = self._insert_buffer(self.zoom_in_buffer, zoom_in_score_tmp)

                zoom_in_score = self._get_zoom_in_score()
                if zoom_in_score > self.last_zoom_in_score:
                    # print("Zoom_in_score:", zoom_in_score)
                    reward += 0.8 * self.dense_reward * zoom_in_score
                    self.last_zoom_in_score = zoom_in_score
                


            if self.fusion:
                # print("fusion")
                if self.concentration.mask is not None:
                    mask = self.concentration.mask
                else:
                    mask = np.zeros((160, 256)) # 160*256

                # 假设obs["rgb"]和mask已经被定义
                rgb = obs["rgb"]  # (3, 160, 256) numpy array
                mask = mask  # (160, 256) numpy array

                # 将mask增加一个维度以匹配rgb的维度，变为(1, 160, 256)
                mask_expanded = np.expand_dims(mask, axis=0)

                # 检查mask是否需要转换类型，与rgb的数据类型保持一致
                if mask_expanded.dtype != rgb.dtype:
                    mask_expanded = mask_expanded.astype(rgb.dtype)

                # 沿着通道维度堆叠rgb数组和扩展后的mask
                obs["rgb"] = np.concatenate((rgb, mask_expanded), axis=0)
            
            

        return obs, reward, done, info
    
    def _get_score(self):
        return np.mean(np.array(self.buffer))

    def _get_zoom_in_score(self):
        return np.mean(np.array(self.zoom_in_buffer))

    def _insert_buffer(self, buffer, logit):
        '''
        向缓冲区中插入新的logit，并保持缓冲区的大小不超过 self.smoothing 指定的大小
        '''
        if buffer is None:
            buffer = [logit]
        elif len(buffer) < self.smoothing:
            buffer.append(logit)
        else:
            buffer = buffer[1:] + [logit]
        return buffer        
    
    
    def _get_slide_window(self):
        # 找出 self.concentration.mask 中值最大的位置
        if self.concentration.mask is not None:
            mask = self.concentration.mask
        else:
            raise ValueError("self.concentration.mask is None, please check your code!")

                
        max_index = np.argmax(mask)
        max_row, max_col = divmod(max_index, 256)

    
    def zoom_in_reward(self):
        pass
        


    

                
