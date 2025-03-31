from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(self, env, clip, prompts=None, dense_reward=.01, clip_target=23, clip_min=21, smoothing=50, **kwargs):
        super().__init__(env)
        self.clip = clip # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        self.clip_target = th.tensor(clip_target)
        self.clip_min = th.tensor(clip_min)
        
        self.buffer = None
        self._clip_state = None, None
        self.last_score = 0

    def reset(self, **kwargs):
        print("==========Now is resetting from ClipWrapper!==========")
        
        self._clip_state = None, self._clip_state[1]
        self.buffer = None
        self.last_score = 0
        print("self.prompt", self.prompt)
        # print("self._clip_state", self._clip_state)
        tmp = self.env.reset(**kwargs)
        print("==========Resetting from ClipWrapper is done!==========")
        return tmp
    
    def step(self, action):
        # print("function step() from clip_wrapper.py")
        obs, reward, done, info = self.env.step(action)

        if len(self.prompt) > 0:
            # print(self.prompt)
            logits, self._clip_state = self.clip.get_logits(obs, self.prompt, self._clip_state)
            logits = logits.detach().cpu()

            self.buffer = self._insert_buffer(self.buffer, logits[:1])
            score = self._get_score()

            if score > self.last_score:
                reward += self.dense_reward * score
                self.last_score = score

        return obs, reward, done, info 

    def _get_score(self):
        '''
        通过将CLIP模型的输出与预设的目标和最小阈值进行比较，计算出一个标准化的得分
        该得分反映了智能体的行为与给定提示的匹配程度
        '''
        return (max(
            th.mean(self.buffer) - self.clip_min,
            0
        ) / (self.clip_target - self.clip_min)).item()

    def _insert_buffer(self, buffer, logits):
        '''
        向缓冲区中插入新的logits，并保持缓冲区的大小不超过 self.smoothing 指定的大小
        '''
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
