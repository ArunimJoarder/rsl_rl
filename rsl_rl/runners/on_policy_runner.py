# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from __future__ import print_function
import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OnPolicyRunner:

	def __init__(self,
				 env: VecEnv,
				 train_cfg,
				 log_dir=None,
				 device='cpu'):

		self.cfg=train_cfg["runner"]
		self.alg_cfg = train_cfg["algorithm"]
		self.policy_cfg = train_cfg["policy"]
		self.device = device
		self.env = env
		
		if self.env.num_agents == None:
			if self.env.num_privileged_obs is not None:
				num_critic_obs = self.env.num_privileged_obs 
			else:
				num_critic_obs = self.env.num_obs
			actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
			actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
															num_critic_obs,
															self.env.num_actions,
															**self.policy_cfg).to(self.device)
			alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
			self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
		else:
			if self.env.num_privileged_obs is not None:
				num_critic_obs = self.env.cfg.env.num_privileged_obs_per_agent 
			else:
				num_critic_obs = self.env.num_obs_per_agent
			self.alg = []
			for k in range(self.env.num_agents):
				actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
				actor_critic: ActorCritic = actor_critic_class( self.env.num_obs_per_agent,
																num_critic_obs,
																self.env.num_actions_per_agent,
																**self.policy_cfg).to(self.device)
				alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
				alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
				self.alg.append(alg)

		self.num_steps_per_env = self.cfg["num_steps_per_env"]
		self.save_interval = self.cfg["save_interval"]

		# init storage and model
		if self.env.num_agents == None:
			self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])
		else:
			for k in range(self.env.num_agents):
				self.alg[k].init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs_per_agent], [self.env.num_privileged_obs_per_agent], [self.env.num_actions_per_agent])


		# Log
		self.log_dir = log_dir
		self.writer = None
		self.tot_timesteps = 0
		self.tot_time = 0
		self.current_learning_iteration = 0

		_, _ = self.env.reset()
	
	def learn(self, num_learning_iterations, init_at_random_ep_len=False):
		# initialize writer
		if self.log_dir is not None and self.writer is None:
			self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
		if init_at_random_ep_len:
			self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
		obs = self.env.get_observations()
		privileged_obs = self.env.get_privileged_observations()
		critic_obs = privileged_obs if privileged_obs is not None else obs
		obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
		if self.env.num_agents == None:
			self.alg.actor_critic.train() # switch to train mode (for dropout for example)
		else:
			for k in range(self.env.num_agents):
				self.alg[k].actor_critic.train()

		ep_infos = []
		rewbuffer = deque(maxlen=100)
		lenbuffer = deque(maxlen=100)
		cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
		cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

		tot_iter = self.current_learning_iteration + num_learning_iterations
		for it in range(self.current_learning_iteration, tot_iter):
			start = time.time()
			# Rollout
			with torch.inference_mode():
				for i in range(self.num_steps_per_env):
					if self.env.num_agents == None:
						actions = self.alg.act(obs, critic_obs)
					else:
						all_actions = []
						for k in range(self.env.num_agents):
							all_actions.append(self.alg[k].act(obs, critic_obs))
						actions = torch.cat(all_actions, dim=1)

					obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
					critic_obs = privileged_obs if privileged_obs is not None else obs
					obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
					
					if self.env.num_agents == None:
						self.alg.process_env_step(rewards, dones, infos)
					else:
						for k in range(self.env.num_agents):
							self.alg[k].process_env_step(rewards[k], dones, infos)


					if self.log_dir is not None:
						# Book keeping
						if 'episode' in infos:
							ep_infos.append(infos['episode'])
						if self.env.num_agents == None:
							cur_reward_sum += rewards
						else:
							for k in range(self.env.num_agents):
								cur_reward_sum += rewards[k]/self.env.num_agents
						cur_episode_length += 1
						new_ids = (dones > 0).nonzero(as_tuple=False)
						rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
						lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
						cur_reward_sum[new_ids] = 0
						cur_episode_length[new_ids] = 0

				stop = time.time()
				collection_time = stop - start

				# Learning step
				start = stop
				if self.env.num_agents == None:
					self.alg.compute_returns(critic_obs)
				else:
					for k in range(self.env.num_agents):
						self.alg[k].compute_returns(critic_obs)
			
			if self.env.num_agents == None:
				mean_value_loss, mean_surrogate_loss = self.alg.update()
			else:
				mean_value_loss, mean_surrogate_loss = 0., 0.
				for k in range(self.env.num_agents):
					mean_value_loss_agent, mean_surrogate_loss_agent = self.alg[k].update()
					mean_value_loss += mean_value_loss_agent
					mean_surrogate_loss += mean_surrogate_loss_agent
				mean_value_loss /= self.env.num_agents
				mean_surrogate_loss /= self.env.num_agents

			stop = time.time()
			learn_time = stop - start
			if self.log_dir is not None:
				self.log(locals())
			if it % self.save_interval == 0:
				self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
			ep_infos.clear()
		
		self.current_learning_iteration += num_learning_iterations
		self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

	def log(self, locs, width=80, pad=35):
		self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
		self.tot_time += locs['collection_time'] + locs['learn_time']
		iteration_time = locs['collection_time'] + locs['learn_time']

		ep_string = f''
		if locs['ep_infos']:
			for key in locs['ep_infos'][0]:
				infotensor = torch.tensor([], device=self.device)
				for ep_info in locs['ep_infos']:
					# handle scalar and zero dimensional tensor infos
					if not isinstance(ep_info[key], torch.Tensor):
						ep_info[key] = torch.Tensor([ep_info[key]])
					if len(ep_info[key].shape) == 0:
						ep_info[key] = ep_info[key].unsqueeze(0)
					infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
				value = torch.mean(infotensor)
				self.writer.add_scalar('Episode/' + key, value, locs['it'])
				ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
		
		if self.env.num_agents == None:
			mean_std = self.alg.actor_critic.std.mean()
		else:
			mean_std = 0.
			for k in range(self.env.num_agents):
				mean_std += self.alg[k].actor_critic.std.mean()
			mean_std /= self.env.num_agents

		fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

		self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
		self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
		if self.env.num_agents == None:
			learning_rate =  self.alg.learning_rate
		else:
			learning_rate = 0.
			for k in range(self.env.num_agents):
				learning_rate_agent = self.alg[k].learning_rate
				self.writer.add_scalar("Loss/learning_rate_agent_" + str(k+1), learning_rate_agent, locs['it'])
				learning_rate += learning_rate_agent
			learning_rate /= self.env.num_agents
		self.writer.add_scalar('Loss/learning_rate', learning_rate, locs['it'])
		self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
		self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
		self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
		self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
		if len(locs['rewbuffer']) > 0:
			self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
			self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
			self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
			self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

		mstr = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

		if len(locs['rewbuffer']) > 0:
			log_string = (f"""{'#' * width}\n"""
						  f"""{mstr.center(width, ' ')}\n\n"""
						  f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
							'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
						  f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
						  f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
						  f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
						  f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
						  f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
						#   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
						#   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
		else:
			log_string = (f"""{'#' * width}\n"""
						  f"""{mstr.center(width, ' ')}\n\n"""
						  f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
							'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
						  f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
						  f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
						  f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
						#   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
						#   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

		log_string += ep_string
		log_string += (f"""{'-' * width}\n"""
					   f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
					   f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
					   f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
					   f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
							   locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
		print(log_string)

	def save(self, path, infos=None):
		if self.env.num_agents is None:
			torch.save({
				'model_state_dict': self.alg.actor_critic.state_dict(),
				'optimizer_state_dict': self.alg.optimizer.state_dict(),
				'iter': self.current_learning_iteration,
				'infos': infos,
				}, path)
		else:
			for k in range(self.env.num_agents):
				path_agent = path[:-3] + "_agent_" + str(k+1) + ".pt"
				torch.save({
					'model_state_dict': self.alg[k].actor_critic.state_dict(),
					'optimizer_state_dict': self.alg[k].optimizer.state_dict(),
					'iter': self.current_learning_iteration,
					'infos': infos,
					}, path_agent)

	def load(self, path, load_optimizer=True):
		loaded_dict = torch.load(path)
		self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
		if load_optimizer:
			self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
		self.current_learning_iteration = loaded_dict['iter']
		return loaded_dict['infos']

	def get_inference_policy(self, device=None):
		self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
		if device is not None:
			self.alg.actor_critic.to(device)
		return self.alg.actor_critic.act_inference
