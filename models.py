import torch
from torch import nn
import numpy as np
from PIL import ImageColor, Image, ImageDraw, ImageFont

import networks
import tools
to_np = lambda x: x.detach().cpu().numpy()


class WorldModel(nn.Module):

  def __init__(self, step, config):
    super(WorldModel, self).__init__()
    self._step = step
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self.encoder_lam = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels)
    self.encoder = networks.ConvEncoder(config.grayscale,
        config.cnn_depth, config.act, config.encoder_kernels)
    if config.size[0] == 64 and config.size[1] == 64:
      embed_size = 2 ** (len(config.encoder_kernels)-1) * config.cnn_depth
      embed_size *= 2 * 2
    else:
      raise NotImplemented(f"{config.size} is not applicable now")
    
    self.embed_size = embed_size
    self.latent_action_net = networks.LatentActionGen(num_embeddings=config.num_latent_action, in_channel=embed_size, embedding_channel=config.latent_action_dim)
    self.mmd = networks.MMDLoss('linear')
    if config.dyn_discrete:
      self.imitation_net = networks.classifier_net(config.dyn_stoch*config.dyn_discrete+config.dyn_deter, (config.dyn_stoch*config.dyn_discrete+config.dyn_deter)//2, config.latent_action_dim)
    else:
      self.imitation_net = networks.classifier_net(config.dyn_stoch+config.dyn_deter, (config.dyn_stoch + config.dyn_deter)//2, config.latent_action_dim)

    self.dynamics_lam = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.latent_action_dim, embed_size, config.device)
    
    self.dynamics = networks.RSSM(
        config.dyn_stoch, config.dyn_deter, config.dyn_hidden,
        config.dyn_input_layers, config.dyn_output_layers,
        config.dyn_rec_depth, config.dyn_shared, config.dyn_discrete,
        config.act, config.dyn_mean_act, config.dyn_std_act,
        config.dyn_temp_post, config.dyn_min_std, config.dyn_cell,
        config.num_actions, embed_size, config.device)
  
    self.heads = nn.ModuleDict()
    channels = (1 if config.grayscale else 3)
    shape = (channels,) + config.size
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.heads['image'] = networks.ConvDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    self.heads['image_latent'] = networks.ConvDecoder(
        feat_size,  # pytorch version
        config.cnn_depth, config.act, shape, config.decoder_kernels,
        config.decoder_thin)
    self.heads['reward'] = networks.DenseHead(
        feat_size,  # pytorch version
        [], config.reward_layers, config.units, config.act)
    if config.pred_discount:
      self.heads['discount'] = networks.DenseHead(
          feat_size,  # pytorch version
          [], config.discount_layers, config.units, config.act, dist='binary')
    for name in config.grad_heads:
      assert name in self.heads, name
    self._model_opt = tools.Optimizer(
        'model', self.parameters(), config.model_lr, config.opt_eps, config.grad_clip,
        config.weight_decay, opt=config.opt,
        use_amp=self._use_amp)
    self._scales = dict(
        reward=config.reward_scale, discount=config.discount_scale)
    
    self.mse_func = torch.nn.MSELoss()
    self.train_source_imitation_mmd_step = config.step_train_wm

  def _train_stop(self, target_dataset, step):
    target_dataset = self.preprocess(target_dataset)

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed = self.encoder(target_dataset)
        post, prior = self.dynamics.observe(embed, target_dataset['action'])
 
    post = {k: v.detach() for k, v in post.items()}
    return post, None, {}

  def _train(self, target_dataset, target_dataset_large, source_dataset, step):
    target_dataset = self.preprocess(target_dataset)
    target_dataset_large = self.preprocess(target_dataset_large)
    # source_dataset = self.preprocess(source_dataset, source=True)

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)

        ########################  train latent wm with taget data
        embed_target = self.encoder_lam(target_dataset_large)
        latent_action_target, _, _, encoding_indices_target = self.latent_action_net(embed_target[:, 1:, :].reshape(-1, self.embed_size), embed_target[:, :-1, :].reshape(-1, self.embed_size))
        encoding_indices_target = encoding_indices_target.reshape(self._config.batch_size, self._config.batch_length_large-1).detach()
        latent_action_target = latent_action_target.reshape(self._config.batch_size, self._config.batch_length_large-1, -1).detach()

        jump_target = torch.ne(encoding_indices_target[:, 1:], encoding_indices_target[:, :-1]).int()
        first_jump_target = torch.ones([encoding_indices_target.shape[0], 1]).to(self._config.device)
        action_jump_target = torch.concat([first_jump_target, jump_target], -1)  
        image_jump_target = torch.cat([action_jump_target[:, 1:], torch.ones(action_jump_target.shape[0], 1).to(self._config.device)], dim=-1) 

        action_mul_target = torch.sort(action_jump_target, dim=1, descending=True)[0][:, :self._config.batch_length]
        image_mul_target = torch.sort(image_jump_target, dim=1, descending=True)[0][:, :self._config.batch_length]

        image_latent_target = target_dataset_large['image'][:, 1:] * image_jump_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        action_latent_target = latent_action_target * action_jump_target.unsqueeze(-1)

        for i in range(image_latent_target.size(0)):
          non_zero_indices_image = torch.nonzero(image_latent_target[i, :, 0, 0, 0], as_tuple=True)[0]  
          non_zero_indices_action = torch.nonzero(action_latent_target[i, :, 0], as_tuple=True)[0]  

          image_latent_target[i] = torch.cat([image_latent_target[i, non_zero_indices_image], torch.zeros_like(image_latent_target[i])], dim=0)[:self._config.batch_length_large-1]
          action_latent_target[i] = torch.cat([action_latent_target[i, non_zero_indices_action], torch.zeros_like(action_latent_target[i])], dim=0)[:self._config.batch_length_large-1]

        image_latent_target = image_latent_target[:, :self._config.batch_length]
        action_latent_target = action_latent_target[:, :self._config.batch_length]

        embed_target_latent = self.encoder(image_latent_target, latent=True)
        post_latent_target, prior_latent_target = self.dynamics_lam.observe(embed_target_latent, action_latent_target)

        kl_loss_target, kl_value_target = self.dynamics_lam.kl_loss(
            post_latent_target, prior_latent_target, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        

        if step <= self.train_source_imitation_mmd_step:
          ########################  train latent wm with source data
          source_dataset = self.preprocess(source_dataset, source=True)
          embed_source = self.encoder_lam(source_dataset)
          latent_action_source, _, _, encoding_indices_source = self.latent_action_net(embed_source[:, 1:, :].reshape(-1, self.embed_size), embed_source[:, :-1, :].reshape(-1, self.embed_size))
          encoding_indices_source = encoding_indices_source.reshape(self._config.batch_size, self._config.batch_length_source-1).detach()
          latent_action_source = latent_action_source.reshape(self._config.batch_size, self._config.batch_length_source-1, -1).detach()

          jump_source = torch.ne(encoding_indices_source[:, 1:], encoding_indices_source[:, :-1]).int()
          first_jump_source = torch.ones([encoding_indices_source.shape[0], 1]).to(self._config.device)
          action_jump_source = torch.concat([first_jump_source, jump_source], -1)  
          image_jump_source = torch.cat([action_jump_source[:, 1:], torch.ones(action_jump_source.shape[0], 1).to(self._config.device)], dim=-1) 

          action_mul_source = torch.sort(action_jump_source, dim=1, descending=True)[0]
          image_mul_source = torch.sort(image_jump_source, dim=1, descending=True)[0]

          image_latent_source = source_dataset['image'][:, 1:] * image_jump_source.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
          action_latent_source = latent_action_source * action_jump_source.unsqueeze(-1)

          for i in range(image_latent_source.size(0)):
            non_zero_indices_image_source = torch.nonzero(image_latent_source[i, :, 0, 0, 0], as_tuple=True)[0]  
            non_zero_indices_action_source = torch.nonzero(action_latent_source[i, :, 0], as_tuple=True)[0]  

            image_latent_source[i] = torch.cat([image_latent_source[i, non_zero_indices_image_source], torch.zeros_like(image_latent_source[i])], dim=0)[:self._config.batch_length_source-1]
            action_latent_source[i] = torch.cat([action_latent_source[i, non_zero_indices_action_source], torch.zeros_like(action_latent_source[i])], dim=0)[:self._config.batch_length_source-1]

          embed_source_latent = self.encoder(image_latent_source, latent=True)
          post_latent_source, prior_latent_source = self.dynamics_lam.observe(embed_source_latent, action_latent_source)

          kl_loss_source, kl_value_source = self.dynamics_lam.kl_loss(
              post_latent_source, prior_latent_source, self._config.kl_forward, kl_balance, kl_free, kl_scale)
          
          ########################  train mmd
          mmd_loss = self.mmd(embed_source_latent.reshape(-1, self.embed_size), embed_target_latent.reshape(-1, self.embed_size))
          
          ########################  train imitation net
          feat_imi_target = self.dynamics_lam.get_feat(post_latent_target)[:, :-1].detach()
          pred_latent_action_target = self.imitation_net(feat_imi_target) 
          pred_latent_action_target = pred_latent_action_target * action_mul_target[:, 1:].unsqueeze(-1)
          imitation_loss_target = self.mse_func(pred_latent_action_target, action_latent_target[:, 1:])

          feat_imi_source = self.dynamics_lam.get_feat(post_latent_source)[:, :-1].detach()
          pred_latent_action_source = self.imitation_net(feat_imi_source) 
          pred_latent_action_source = pred_latent_action_source * action_mul_source[:, 1:].unsqueeze(-1)
          imitation_loss_source = self.mse_func(pred_latent_action_source, action_latent_source[:, 1:])

        ########################  train true wm with taget data
        embed = self.encoder(target_dataset)
        post, prior = self.dynamics.observe(embed, target_dataset['action'])
        kl_loss, kl_value = self.dynamics.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)

        losses = {}
        for name, head in self.heads.items():
          if name == 'image':
            feat = self.dynamics.get_feat(post)
            pred = head(feat)
            like = pred.log_prob(target_dataset[name])
            losses['image'] = -torch.mean(like)
          if name == 'image_latent':
            feat_target = self.dynamics_lam.get_feat(post_latent_target)
            pred_target = head(feat_target) 
            like_target = pred_target.log_prob(image_latent_target) * image_mul_target
            losses['image_target'] = -torch.mean(like_target)
            
            if step <= self.train_source_imitation_mmd_step:
              feat_source = self.dynamics_lam.get_feat(post_latent_source)
              pred_source = head(feat_source)
              like_source = pred_source.log_prob(image_latent_source) * image_mul_source
              losses['image_source'] = -torch.mean(like_source)
          if name == 'reward':
            feat = self.dynamics.get_feat(post)
            pred = head(feat)
            like = pred.log_prob(target_dataset[name])
            losses[name] = -torch.mean(like)
        if step <= self.train_source_imitation_mmd_step:
          model_loss = sum(losses.values()) + kl_loss + kl_loss_target + kl_loss_source + imitation_loss_target + imitation_loss_source + mmd_loss * 5
        else: 
          model_loss = sum(losses.values()) + kl_loss + kl_loss_target
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'{name}_loss': to_np(loss) for name, loss in losses.items()})
    metrics['kl_balance'] = kl_balance
    metrics['kl_free'] = kl_free
    metrics['kl_scale'] = kl_scale
    metrics['kl_loss_target'] = to_np(kl_loss_target)
    if step <= self.train_source_imitation_mmd_step:
      metrics['kl_loss_source'] = to_np(kl_loss_source)
      metrics['imitation_loss_target'] = to_np(imitation_loss_target)
      metrics['imitation_loss_source'] = to_np(imitation_loss_source)
      metrics['mmd_loss'] = to_np(mmd_loss)
    metrics['kl'] = to_np(torch.mean(kl_value))
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics.get_dist(post).entropy()))
      context = dict(
          embed=embed, feat=self.dynamics.get_feat(post),
          kl=kl_value, postent=self.dynamics.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics  

  def _train_lam(self, target_dataset, source_dataset):
    target_dataset = self.preprocess(target_dataset)
    source_dataset = self.preprocess(source_dataset, source=True)

    with tools.RequiresGrad(self):
      with torch.cuda.amp.autocast(self._use_amp):
        embed_target = self.encoder_lam(target_dataset)
        embed_source = self.encoder_lam(source_dataset)

        latent_action, vq_loss, _, encoding_indices = self.latent_action_net(embed_target[:, 1:, :].reshape(-1, self.embed_size), embed_target[:, :-1, :].reshape(-1, self.embed_size))
        vq_loss = torch.mean(vq_loss)
        latent_action = latent_action.reshape(self._config.batch_size, self._config.batch_length-1, -1)

        mmd_loss = self.mmd(embed_source.reshape(-1, self.embed_size), embed_target.reshape(-1, self.embed_size))

        post, prior = self.dynamics_lam.observe(embed_target[:, 1:], latent_action)
        kl_balance = tools.schedule(self._config.kl_balance, self._step)
        kl_free = tools.schedule(self._config.kl_free, self._step)
        kl_scale = tools.schedule(self._config.kl_scale, self._step)
        kl_loss, kl_value = self.dynamics_lam.kl_loss(
            post, prior, self._config.kl_forward, kl_balance, kl_free, kl_scale)
        losses = {}
        likes = {}
        for name, head in self.heads.items():
          if name == 'reward' or name == 'image_latent':
            continue
          grad_head = (name in self._config.grad_heads)
          feat = self.dynamics_lam.get_feat(post)
          feat = feat if grad_head else feat.detach()
          pred = head(feat)
          like = pred.log_prob(target_dataset[name][:,1:])
          likes[name] = like
          losses[name] = -torch.mean(like) * self._scales.get(name, 1.0)
        model_loss = sum(losses.values()) + kl_loss + vq_loss + mmd_loss * 5
      metrics = self._model_opt(model_loss, self.parameters())

    metrics.update({f'lam_{name}_loss': to_np(loss) for name, loss in losses.items()})
    # metrics['kl_balance'] = kl_balance
    # metrics['kl_free'] = kl_free
    # metrics['kl_scale'] = kl_scale
    metrics['lam_kl'] = to_np(torch.mean(kl_value))
    metrics['lam_vq'] = to_np(vq_loss)
    metrics['lam_mmd_loss'] = to_np(mmd_loss)
    with torch.cuda.amp.autocast(self._use_amp):
      metrics['prior_ent'] = to_np(torch.mean(self.dynamics_lam.get_dist(prior).entropy()))
      metrics['post_ent'] = to_np(torch.mean(self.dynamics_lam.get_dist(post).entropy()))
      context = dict(
          embed=embed_target, feat=self.dynamics_lam.get_feat(post),
          kl=kl_value, postent=self.dynamics_lam.get_dist(post).entropy())
    post = {k: v.detach() for k, v in post.items()}
    return post, context, metrics

  def preprocess(self, obs, source=False):
    obs = obs.copy()
    obs['image'] = torch.Tensor(obs['image']) / 255.0 - 0.5
    if not source:
      if self._config.clip_rewards == 'tanh':
        obs['reward'] = torch.tanh(torch.Tensor(obs['reward'])).unsqueeze(-1)
      elif self._config.clip_rewards == 'identity':
        obs['reward'] = torch.Tensor(obs['reward']).unsqueeze(-1)
      else:
        raise NotImplemented(f'{self._config.clip_rewards} is not implemented')
    if 'discount' in obs:
      obs['discount'] *= self._config.discount
      obs['discount'] = torch.Tensor(obs['discount']).unsqueeze(-1)
    obs = {k: torch.Tensor(v).to(self._config.device) for k, v in obs.items()}
    return obs

  def video_pred(self, data):
    data = self.preprocess(data)
    truth = data['image'][:6] + 0.5
    embed = self.encoder(data)

    states, _ = self.dynamics.observe(embed[:6, :5], data['action'][:6, :5])
    recon = self.heads['image'](
        self.dynamics.get_feat(states)).mode()[:6]
    reward_post = self.heads['reward'](
        self.dynamics.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics.imagine(data['action'][:6, 5:], init)

    futher_feat = torch.zeros_like(self.dynamics.get_feat(prior))
    for i in range((self._config.batch_length-5)//5):
      latent = {}
      latent['stoch'] = prior['stoch'][:, i*5, :]
      latent['deter'] = prior['deter'][:, i*5, :]
      feat_ = self.dynamics.get_feat(latent)
      latent_action_cur = self.imitation_net(feat_)
      distances = (torch.sum(latent_action_cur ** 2, dim=1, keepdim=True)
                      + torch.sum(self.latent_action_net.quantizer.embedding.weight ** 2, dim=1)
                      - 2 * torch.matmul(latent_action_cur, self.latent_action_net.quantizer.embedding.weight.t()))
      encoding_indices_mid = torch.argmin(distances, dim=1)
      encoding_indices = encoding_indices_mid.unsqueeze(1)
      encodings = torch.zeros(encoding_indices.shape[0], self._config.num_latent_action).to(self._config.device)
      encodings = torch.scatter(encodings, 1, encoding_indices, 1)
      quantized = torch.matmul(encodings, self.latent_action_net.quantizer.embedding.weight)
      quantized = latent_action_cur + (quantized - latent_action_cur).detach()
      succ_future = self.dynamics_lam.img_step(latent, quantized, sample=self._config.imag_sample)
      future = self.dynamics_lam.get_feat(succ_future)
      futher_feat[:, i*5:(i+1)*5, :] = future.unsqueeze(1).repeat(1,5,1)
    openl_future = self.heads['image'](futher_feat).mode()
    model_future = torch.cat([recon[:, :5] + 0.5, openl_future + 0.5], 1)

    openl = self.heads['image'](self.dynamics.get_feat(prior)).mode()
    reward_prior = self.heads['reward'](self.dynamics.get_feat(prior)).mode()
    model = torch.cat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, model_future, error], 2)
  
  def video_pred_lam(self, data, source=False):
    data = self.preprocess(data, source=source)

    truth = data['image'][:6, 1:] + 0.5
    embed = self.encoder_lam(data)
    latent_action, _, _, latent_indices = self.latent_action_net(embed[:, 1:, :].reshape(-1, self.embed_size), embed[:, :-1, :].reshape(-1, self.embed_size))
    if source == True:
      length = self._config.batch_length_source
    else:
      length = self._config.batch_length
    latent_action = latent_action.reshape(self._config.batch_size, length-1, -1)

    states, _ = self.dynamics_lam.observe(embed[:6, 1:5], latent_action[:6, :4])
    
    recon = self.heads['image'](self.dynamics_lam.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics_lam.imagine(latent_action[:6, 4:], init)
    openl = self.heads['image'](self.dynamics_lam.get_feat(prior)).mode()
    model = torch.cat([recon[:, :4] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)

  def video_pred_latent(self, data, source=False): 
    data = self.preprocess(data, source=source)
    if source == True:
      length = self._config.batch_length_source
    else:
      length = self._config.batch_length_large

    embed_source = self.encoder_lam(data)
    latent_action_source, _, _, encoding_indices_source = self.latent_action_net(embed_source[:, 1:, :].reshape(-1, self.embed_size), embed_source[:, :-1, :].reshape(-1, self.embed_size))
    encoding_indices_source = encoding_indices_source.reshape(self._config.batch_size, length-1).detach()
    latent_action_source = latent_action_source.reshape(self._config.batch_size, length-1, -1).detach()

    jump_source = torch.ne(encoding_indices_source[:, 1:], encoding_indices_source[:, :-1]).int()
    first_jump_source = torch.ones([encoding_indices_source.shape[0], 1]).to(self._config.device)
    action_jump_source = torch.concat([first_jump_source, jump_source], -1)  
    image_jump_source = torch.cat([action_jump_source[:, 1:], torch.ones(action_jump_source.shape[0], 1).to(self._config.device)], dim=-1) 

    image_latent_source = data['image'][:, 1:] * image_jump_source.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    action_latent_source = latent_action_source * action_jump_source.unsqueeze(-1)

    for i in range(image_latent_source.size(0)):
      non_zero_indices_image_source = torch.nonzero(image_latent_source[i, :, 0, 0, 0], as_tuple=True)[0]  
      non_zero_indices_action_source = torch.nonzero(action_latent_source[i, :, 0], as_tuple=True)[0]  

      image_latent_source[i] = torch.cat([image_latent_source[i, non_zero_indices_image_source], torch.zeros_like(image_latent_source[i])], dim=0)[:length-1]
      action_latent_source[i] = torch.cat([action_latent_source[i, non_zero_indices_action_source], torch.zeros_like(action_latent_source[i])], dim=0)[:length-1]

    truth = image_latent_source[:6, :] + 0.5
    embed = self.encoder(image_latent_source, latent=True)

    states, _ = self.dynamics_lam.observe(embed[:6, :4], action_latent_source[:6, :4])
    
    recon = self.heads['image_latent'](self.dynamics_lam.get_feat(states)).mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.dynamics_lam.imagine(action_latent_source[:6, 4:], init)
    openl = self.heads['image_latent'](self.dynamics_lam.get_feat(prior)).mode()
    model = torch.cat([recon[:, :4] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2

    return torch.cat([truth, model, error], 2)


class ImagBehavior(nn.Module):

  def __init__(self, config, world_model, stop_grad_actor=True, reward=None):
    super(ImagBehavior, self).__init__()
    self._use_amp = True if config.precision==16 else False
    self._config = config
    self._world_model = world_model
    self._stop_grad_actor = stop_grad_actor
    self._reward = reward
    if config.dyn_discrete:
      feat_size = config.dyn_stoch * config.dyn_discrete + config.dyn_deter
    else:
      feat_size = config.dyn_stoch + config.dyn_deter
    self.actor = networks.ActionHead(
        feat_size+config.latent_action_dim,  # pytorch version  +config.latent_action_dim
        config.num_actions, config.actor_layers, config.units, config.act,
        config.actor_dist, config.actor_init_std, config.actor_min_std,
        config.actor_dist, config.actor_temp, config.actor_outscale)
    self.value = networks.DenseHead(
        feat_size+config.latent_action_dim,  # pytorch version
        [], config.value_layers, config.units, config.act,
        config.value_head)
    if config.slow_value_target or config.slow_actor_target:
      self._slow_value = networks.DenseHead(
          feat_size+config.latent_action_dim,  # pytorch version
          [], config.value_layers, config.units, config.act)
      self._updates = 0
    kw = dict(wd=config.weight_decay, opt=config.opt, use_amp=self._use_amp)
    self._actor_opt = tools.Optimizer(
        'actor', self.actor.parameters(), config.actor_lr, config.opt_eps, config.actor_grad_clip,
        **kw)
    self._value_opt = tools.Optimizer(
        'value', self.value.parameters(), config.value_lr, config.opt_eps, config.value_grad_clip,
        **kw)
    self.zero = torch.zeros(1, config.batch_size*config.batch_length).to(self._config.device)

  def _train(
      self, start, objective=None, action=None, reward=None, imagine=None, tape=None, repeats=None):
    objective = objective or self._reward
    self._update_slow_target()
    metrics = {}

    with tools.RequiresGrad(self.actor):  
      with torch.cuda.amp.autocast(self._use_amp):
        imag_feat, imag_state, imag_action, feat_ori, feat_future = self._imagine(
            start, self.actor, self._config.imag_horizon, repeats)

        pos_dist = torch.cat([self.zero, torch.norm(feat_ori[1:] - feat_future[:-1], p=2, dim=-1)], dim=0).unsqueeze(-1)
        int_reward = - pos_dist
        reward = objective(imag_feat, imag_state, imag_action) + int_reward
        actor_ent = self.actor(imag_feat).entropy()
        state_ent = self._world_model.dynamics.get_dist(
            imag_state).entropy()
        target, weights = self._compute_target(
            imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
            self._config.slow_actor_target)
        actor_loss, mets = self._compute_actor_loss(
            imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
            weights)
        metrics.update(mets)
        if self._config.slow_value_target != self._config.slow_actor_target:
          target, weights = self._compute_target(
              imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
              self._config.slow_value_target)
        value_input = imag_feat

    with tools.RequiresGrad(self.value):
      with torch.cuda.amp.autocast(self._use_amp):
        value = self.value(value_input[:-1].detach())
        target = torch.stack(target, dim=1)
        value_loss = -value.log_prob(target.detach())
        if self._config.value_decay:
          value_loss += self._config.value_decay * value.mode()
        value_loss = torch.mean(weights[:-1] * value_loss[:,:,None])

    metrics['reward_mean'] = to_np(torch.mean(reward))
    metrics['reward_std'] = to_np(torch.std(reward))
    metrics['actor_ent'] = to_np(torch.mean(actor_ent))
    with tools.RequiresGrad(self):
      metrics.update(self._actor_opt(actor_loss, self.actor.parameters()))
      metrics.update(self._value_opt(value_loss, self.value.parameters()))
    return imag_feat, imag_state, imag_action, weights, metrics

  def _imagine(self, start, policy, horizon, repeats=None):
    dynamics = self._world_model.dynamics
    dynamics_lam = self._world_model.dynamics_lam
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")
    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    start = {k: flatten(v) for k, v in start.items()}
    def step(prev, _):
      state, _, _, _, _ = prev
      feat_ori = dynamics.get_feat(state)

      latent_action_cur = self._world_model.imitation_net(feat_ori)
      distances = (torch.sum(latent_action_cur ** 2, dim=1, keepdim=True)
                      + torch.sum(self._world_model.latent_action_net.quantizer.embedding.weight ** 2, dim=1)
                      - 2 * torch.matmul(latent_action_cur, self._world_model.latent_action_net.quantizer.embedding.weight.t()))
      encoding_indices_mid = torch.argmin(distances, dim=1)
      encoding_indices = encoding_indices_mid.unsqueeze(1)
      encodings = torch.zeros(encoding_indices.shape[0], self._config.num_latent_action).to(self._config.device)
      encodings = torch.scatter(encodings, 1, encoding_indices, 1)
      quantized = torch.matmul(encodings, self._world_model.latent_action_net.quantizer.embedding.weight)
      quantized = latent_action_cur + (quantized - latent_action_cur).detach()
      succ_future = dynamics_lam.img_step(state, quantized, sample=self._config.imag_sample)
      self.feat_future = dynamics_lam.get_feat(succ_future)

      feat = torch.concat([feat_ori, quantized], -1)
      inp = feat.detach() if self._stop_grad_actor else feat
      action = policy(inp).sample()
      succ = dynamics.img_step(state, action, sample=self._config.imag_sample)
      return succ, feat, action, feat_ori, self.feat_future
    feat = 0 * dynamics.get_feat(start)
    # action = policy(feat).mode()
    succ, feats, actions, feat_ori, feat_future = tools.static_scan(
        step, [torch.arange(horizon)], (start, None, None, None, None))
    states = {k: torch.cat([
        start[k][None], v[:-1]], 0) for k, v in succ.items()}
    if repeats:
      raise NotImplemented("repeats is not implemented in this version")

    return feats, states, actions, feat_ori, feat_future

  def _compute_target(
      self, imag_feat, imag_state, imag_action, reward, actor_ent, state_ent,
      slow):
    if 'discount' in self._world_model.heads:
      inp = self._world_model.dynamics.get_feat(imag_state)
      discount = self._world_model.heads['discount'](inp).mean
    else:
      discount = self._config.discount * torch.ones_like(reward)
    if self._config.future_entropy and self._config.actor_entropy() > 0:
      reward += self._config.actor_entropy() * actor_ent
    if self._config.future_entropy and self._config.actor_state_entropy() > 0:
      reward += self._config.actor_state_entropy() * state_ent
    if slow:
      value = self._slow_value(imag_feat).mode()
    else:
      value = self.value(imag_feat).mode()
    target = tools.lambda_return(
        reward[:-1], value[:-1], discount[:-1],
        bootstrap=value[-1], lambda_=self._config.discount_lambda, axis=0)
    weights = torch.cumprod(
        torch.cat([torch.ones_like(discount[:1]), discount[:-1]], 0), 0).detach()
    return target, weights

  def _compute_actor_loss(
      self, imag_feat, imag_state, imag_action, target, actor_ent, state_ent,
      weights):
    metrics = {}
    inp = imag_feat.detach() if self._stop_grad_actor else imag_feat
    policy = self.actor(inp)
    actor_ent = policy.entropy()
    target = torch.stack(target, dim=1)
    if self._config.imag_gradient == 'dynamics':
      actor_target = target
    elif self._config.imag_gradient == 'reinforce':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
    elif self._config.imag_gradient == 'both':
      actor_target = policy.log_prob(imag_action)[:-1][:, :, None] * (
          target - self.value(imag_feat[:-1]).mode()).detach()
      mix = self._config.imag_gradient_mix()
      actor_target = mix * target + (1 - mix) * actor_target
      metrics['imag_gradient_mix'] = mix
    else:
      raise NotImplementedError(self._config.imag_gradient)
    if not self._config.future_entropy and (self._config.actor_entropy() > 0):
      actor_target += self._config.actor_entropy() * actor_ent[:-1][:,:,None]
    if not self._config.future_entropy and (self._config.actor_state_entropy() > 0):
      actor_target += self._config.actor_state_entropy() * state_ent[:-1]
    actor_loss = -torch.mean(weights[:-1] * actor_target)
    return actor_loss, metrics

  def _update_slow_target(self):
    if self._config.slow_value_target or self._config.slow_actor_target:
      if self._updates % self._config.slow_target_update == 0:
        mix = self._config.slow_target_fraction
        for s, d in zip(self.value.parameters(), self._slow_value.parameters()):
          d.data = mix * s.data + (1 - mix) * d.data
      self._updates += 1


