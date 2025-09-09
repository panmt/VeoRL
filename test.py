import argparse
import collections
import functools
import os
import pathlib
import sys 
import warnings
import random
import torch.nn.functional as F
from datetime import datetime
from agents.navigation.carla_env_dream import CarlaEnv

os.environ['MUJOCO_GL'] = 'egl' 

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import wrappers

import torch
from torch import nn
from torch import distributions as torchd
to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):

  def __init__(self, config, logger, target_dataset, source_dataset):
    super(Dreamer, self).__init__()
    self._config = config
    self._logger = logger
    self._should_log = tools.Every(config.log_every)
    self._should_train = tools.Every(config.train_every)
    self._should_pretrain = tools.Once()
    self._should_reset = tools.Every(config.reset_every)
    self._should_expl = tools.Until(int(
        config.expl_until / config.action_repeat))
    self._metrics = {}
    self._step = count_steps(config.traindir)
    # Schedules.
    config.actor_entropy = (
        lambda x=config.actor_entropy: tools.schedule(x, self._step))
    config.actor_state_entropy = (
        lambda x=config.actor_state_entropy: tools.schedule(x, self._step))
    config.imag_gradient_mix = (
        lambda x=config.imag_gradient_mix: tools.schedule(x, self._step))
    self._target_dataset = target_dataset
    self._source_dataset = source_dataset
    self._wm = models.WorldModel(self._step, config)
    self._task_behavior = models.ImagBehavior(
        config, self._wm, config.behavior_stop_grad)
    reward = lambda f, s, a: self._wm.heads['reward'](f).mean
    self._expl_behavior = dict(
        greedy=lambda: self._task_behavior,
        random=lambda: expl.Random(config),
        plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
    )[config.expl_behavior]()

  def __call__(self, obs, reset, state=None, reward=None, training=True):
    step = self._step
    if self._should_reset(step):
      state = None
    if state is not None and reset.any():
      mask = 1 - reset
      for key in state[0].keys():
        for i in range(state[0][key].shape[0]):
          state[0][key][i] *= mask[i]
      for i in range(len(state[1])):
        state[1][i] *= mask[i]
    if training and self._should_train(step):
      steps = (
          self._config.pretrain if self._should_pretrain()
          else self._config.train_steps)
      for _ in range(steps):
        self._train(next(self._dataset))
      if self._should_log(step):
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        openl = self._wm.video_pred(next(self._dataset))
        self._logger.video('train_openl', to_np(openl))
        self._logger.write(fps=True)

    policy_output, state = self._policy(obs, state, training)

    if training:
      self._step += len(reset)
      self._logger.step = self._config.action_repeat * self._step
    return policy_output, state

  def _policy(self, obs, state, training):
    if state is None:
      batch_size = len(obs['image'])
      latent = self._wm.dynamics.initial(len(obs['image']))
      action = torch.zeros((batch_size, self._config.num_actions)).to(self._config.device)
      self.roll_step = 0
    else:
      latent, action = state
    embed = self._wm.encoder(self._wm.preprocess(obs))
    latent, _ = self._wm.dynamics.obs_step(
        latent, action, embed, self._config.collect_dyn_sample)
    if self._config.eval_state_mean:
      latent['stoch'] = latent['mean']
    feat = self._wm.dynamics.get_feat(latent)

    latent_action_cur = self._wm.imitation_net(feat)
    distances = (torch.sum(latent_action_cur ** 2, dim=1, keepdim=True)
                    + torch.sum(self._wm.latent_action_net.quantizer.embedding.weight ** 2, dim=1)
                    - 2 * torch.matmul(latent_action_cur, self._wm.latent_action_net.quantizer.embedding.weight.t()))
    encoding_indices_mid = torch.argmin(distances, dim=1)
    encoding_indices = encoding_indices_mid.unsqueeze(1)
    encodings = torch.zeros(encoding_indices.shape[0], self._config.num_latent_action).to(self._config.device)
    encodings = torch.scatter(encodings, 1, encoding_indices, 1)
    quantized = torch.matmul(encodings, self._wm.latent_action_net.quantizer.embedding.weight)
    quantized = latent_action_cur + (quantized - latent_action_cur).detach()
    feat = torch.concat([feat, quantized], -1)

    if not training:
      actor = self._task_behavior.actor(feat)
      action = actor.mode()
    elif self._should_expl(self._step):
      actor = self._expl_behavior.actor(feat)
      action = actor.sample()
    else:
      actor = self._task_behavior.actor(feat)
      action = actor.sample()
    logprob = actor.log_prob(action)
    latent = {k: v.detach()  for k, v in latent.items()}
    action = action.detach()
    if self._config.actor_dist == 'onehot_gumble':
      action = torch.one_hot(torch.argmax(action, dim=-1), self._config.num_actions)
    action = self._exploration(action, training)
    policy_output = {'action': action, 'logprob': logprob}
    state = (latent, action)
    return policy_output, state

  def _exploration(self, action, training):
    amount = self._config.expl_amount if training else self._config.eval_noise
    if amount == 0:
      return action
    if 'onehot' in self._config.actor_dist:
      probs = amount / self._config.num_actions + (1 - amount) * action
      exit()
      return tools.OneHotDist(probs=probs).sample()
    else:
      exit()
      return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
    raise NotImplementedError(self._config.action_noise)

  def _train(self, target_dataset, target_dataset_large, source_dataset, step):
    metrics = {}
    if step == (self._config.step_train_ban + 1):
      self._wm.encoder.load_state_dict(self._wm.encoder_lam.state_dict())
      self._wm.heads['image_latent'].load_state_dict(self._wm.heads['image'].state_dict())
    if step <= self._config.step_train_ban:
      post, context, mets = self._wm._train_lam(target_dataset, source_dataset)
      metrics.update(mets)
    elif step > self._config.step_train_ban and step <= self._config.step_train_wm:
      post, context, mets = self._wm._train(target_dataset, target_dataset_large, source_dataset, step)
      metrics.update(mets)
    else:
      post, context, mets = self._wm._train_stop(target_dataset, step)
      metrics.update(mets)
      start = post
      reward = lambda f, s, a: self._wm.heads['reward'](
          self._wm.dynamics.get_feat(s)).mode()
      metrics.update(self._task_behavior._train(start, reward)[-1])
    for name, value in metrics.items():
      if not name in self._metrics.keys():
        self._metrics[name] = [value]
      else:
        self._metrics[name].append(value)
    if step % self._config.log_every == 0:  #### 1000
        for name, values in self._metrics.items():
          self._logger.scalar(name, float(np.mean(values)))
          self._metrics[name] = []
        if step <= self._config.step_train_ban:
          openl = self._wm.video_pred_lam(target_dataset)
          self._logger.video('train_openl_lam', to_np(openl))
        else:
          openl = self._wm.video_pred(target_dataset)
          self._logger.video('train_openl', to_np(openl))
          if step <= self._config.step_train_wm:
            openl_target = self._wm.video_pred_latent(target_dataset_large)
            self._logger.video('train_openl_latent_target', to_np(openl_target))
            openl_source = self._wm.video_pred_latent(source_dataset, source=True)
            self._logger.video('train_openl_latent_source', to_np(openl_source))
        self._logger.write(fps=True)

def count_steps(folder):
  return sum(int(str(n).split('-')[-1][:-4]) - 1 for n in folder.glob('*.npz'))


def make_dataset(episodes, config, index=None):
  if index == 'source':
    length = config.batch_length_source
  elif index == 'target':
    length = config.batch_length
  elif index == 'target_large':
    length = config.batch_length_large
  else:
    length = config.batch_length

  generator = tools.sample_episodes(
      episodes, length, config.oversample_ends)
  dataset = tools.from_generator(generator, config.batch_size)
  return dataset


def make_env(config, logger, mode, train_eps, eval_eps):
  suite, task = config.task.split('_', 1)
  if suite == 'dmc':
    env = wrappers.DeepMindControl(task, config.action_repeat, config.size)
    env = wrappers.NormalizeActions(env)
  elif suite == 'atari':
    env = wrappers.Atari(
        task, config.action_repeat, config.size,
        grayscale=config.grayscale,
        life_done=False and ('train' in mode),
        sticky_actions=True,
        all_actions=True)
    env = wrappers.OneHotAction(env)
  elif suite == 'dmlab':
    env = wrappers.DeepMindLabyrinth(
        task,
        mode if 'train' in mode else 'test',
        config.action_repeat)
    env = wrappers.OneHotAction(env)
  elif suite == "metaworld":
      task = "-".join(task.split("_"))
      env = wrappers.MetaWorld(
          task,
          config.seed,
          config.action_repeat,
          config.size,
          config.camera,
      )
      env = wrappers.NormalizeActions(env)
  elif suite == 'carla':
    env = CarlaEnv(
            render_display=False,  # for local debugging only
            display_text=False,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=config.image_size,
            max_episode_steps=1000,
            frame_skip=config.action_repeat,
            is_other_cars=True,
            port=2000
        )
  elif suite == 'minedojo':
    import envs.minedojo as minedojo
    log_dir = os.path.join(config.logdir, config.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    kwargs=dict(
      log_dir=log_dir,
    )
    env = minedojo.make_env(task, **kwargs)
    env = wrappers.OneHotAction(env)
  else:
    raise NotImplementedError(suite)
  env = wrappers.TimeLimit(env, config.time_limit)
  env = wrappers.SelectAction(env, key='action')
  callbacks = [functools.partial(
      process_episode, config, logger, mode, train_eps, eval_eps)]
  if suite == 'carla':
    env = wrappers.CollectDataset_Carla(env, callbacks, logger=logger, mode=mode, eval_num=config.eval_num)
  else:
    env = wrappers.CollectDataset(env, callbacks, logger=logger, mode=mode, eval_num=config.eval_num)
  env = wrappers.RewardObs(env)
  return env


def process_episode(config, logger, mode, train_eps, eval_eps, episode):
  directory = dict(train=config.traindir, eval=config.evaldir)[mode]
  cache = dict(train=train_eps, eval=eval_eps)[mode]
  filename = tools.save_episodes(directory, [episode])[0]
  length = len(episode['reward']) - 1
  score = float(episode['reward'].astype(np.float64).sum())
  video = episode['image']
  if mode == 'eval':
    cache.clear()
  if mode == 'train' and config.dataset_size:
    total = 0
    for key, ep in reversed(sorted(cache.items(), key=lambda x: x[0])):
      if total <= config.dataset_size - length:
        total += len(ep['reward']) - 1
      else:
        del cache[key]
    logger.scalar('dataset_size', total + length)
  cache[str(filename)] = episode
  print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
  logger.scalar(f'{mode}_return', score)
  logger.scalar(f'{mode}_length', length)
  logger.scalar(f'{mode}_episodes', len(cache))
  if mode == 'eval' or config.expl_gifs:
    logger.video(f'{mode}_policy', video[None])
  logger.write()


def main(config):
  logdir = pathlib.Path(config.logdir).expanduser()
  config.traindir = config.traindir or logdir / 'train_eps'
  config.evaldir = config.evaldir or logdir / 'eval_eps'
  config.steps //= config.action_repeat
  config.eval_every //= config.action_repeat
  config.log_every //= config.action_repeat
  config.time_limit //= config.action_repeat
  config.act = getattr(torch.nn, config.act)

  print('Logdir', logdir)
  logdir.mkdir(parents=True, exist_ok=True)
  config.traindir.mkdir(parents=True, exist_ok=True)
  config.evaldir.mkdir(parents=True, exist_ok=True)
  step = 0
  logger = tools.Logger(logdir, step)

  print('Create envs.')
  directory = config.traindir
  train_eps = tools.load_episodes(directory, limit=config.dataset_size)
  directory = config.evaldir
  eval_eps = tools.load_episodes(directory, limit=1)
  make = lambda mode: make_env(config, logger, mode, train_eps, eval_eps)
  suite, task = config.task.split('_', 1)
  eval_envs = [make('eval') for _ in range(config.envs)]
  acts = eval_envs[0].action_space
  if suite == 'carla':
    config.num_actions = acts.shape[0]
  else:
    config.num_actions = acts.n if hasattr(acts, 'n') else acts.shape[0]

  print('Simulate agent.')
  eval_dataset = make_dataset(eval_eps, config)

  agent = Dreamer(config, logger, None, None).to(config.device)
  agent.requires_grad_(requires_grad=False)
  
  agent.load_state_dict(torch.load(config.eval_pretrained_model))
  agent._should_pretrain._once = False

  for i in range(10):
    logger.write()
    print('Start evaluation.')
    eval_policy = functools.partial(agent, training=False)
    tools.simulate(eval_policy, eval_envs, episodes=config.eval_num)  # config.eval_num
    video_pred = agent._wm.video_pred(next(eval_dataset))
    logger.video('eval_openl', to_np(video_pred))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--configs', nargs='+', required=True)
  args, remaining = parser.parse_known_args()
  configs = yaml.safe_load(
      (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())
  defaults = {}
  for name in args.configs:
    defaults.update(configs[name])
  parser = argparse.ArgumentParser()
  for key, value in sorted(defaults.items(), key=lambda x: x[0]):
    arg_type = tools.args_type(value)
    parser.add_argument(f'--{key}', type=arg_type, default=arg_type(value))
  main(parser.parse_args(remaining))
