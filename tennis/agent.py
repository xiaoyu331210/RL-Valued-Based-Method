import model
from utils import OUNoise, ReplayBufferMultiAgent, ReplayBuffer, ReplayBufferMAPPO, soft_update
import numpy as np
from dataclasses import dataclass
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

@dataclass
class DDPGConfig:
    num_state: int = 0
    num_action: int = 0
    num_agent: int = 1
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    batch_size: int = 128
    update_every_timestamp: int = 20
    update_time_each_stamp: int = 10
    discount_factor: float = 0.99
    replay_buffer_size: int = 1e6
    weight_decay: float = 0

class DDPGAgent():
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        # declare the network
        self.actor = model.Actor(config.num_state, config.num_action).to(self.device)
        self.actor_target = model.Actor(config.num_state, config.num_action).to(self.device)

        self.critic = model.Critic(config.num_state, config.num_action).to(self.device)
        self.critic_target = model.Critic(config.num_state, config.num_action).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_learning_rate)

        self.noise = OUNoise((config.num_agent, config.num_action), 2)

        # experience replay queue
        self.memory = ReplayBuffer(int(config.replay_buffer_size), config.batch_size, self.device)

        # record time steps
        self.t_step = 0

        # model weight file name
        self.default_actor_weight_prefix = '/ddpg_actor'
        self.default_critic_weight_prefix = '/ddpg_critic'
        self.default_actor_target_weight_prefix = '/ddpg_actor_target'
        self.default_critic_target_weight_prefix = '/ddpg_critic_target'

    def reset(self):
        self.noise.reset()

    def act(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        self.actor.eval()
        with torch.no_grad():
            actions = self.actor(states).cpu().data.numpy()
        self.actor.train()

        # return actions
        actions += self.noise.sample()
        return np.clip(actions, -1., 1.)


    def step(self, state, action, reward, next_state, done):
        """add one observation tuple into replay buffer, and optimize the model

        Params
        ======
            state (array_like): the current state
            action (array_like): the current action taken
            reward (array_like): the reward after taking the current action
            next_state (array_like): the next state after taking action
            done (array_like): whether the episode is finished
        """
        # Update the memory with the latest experience, and perform learn step
        # self.memory.add(state, action, reward, next_state, done)
        for i in range(self.config.num_agent):
            self.memory.add(state[i,:], action[i,:], reward[i], next_state[i,:], done[i])

        if len(self.memory) < self.config.batch_size:
            return False
        self.t_step = (self.t_step + 1) % self.config.update_every_timestamp
        if 0 != self.t_step:
            return False

        for _ in range(self.config.update_time_each_stamp):
            experiences = self.memory.sample()
            self.__learn(experiences, self.config.discount_factor)
        return True

    def __learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences
        # ---------------------------- update critic ---------------------------- #
        next_actions = self.actor_target(next_states)
        # compute the Q value from the next states
        next_state_Q = self.critic_target(next_states, next_actions)
        q_targets = rewards + (discount_factor * next_state_Q * (1- dones))
        q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(q_targets, q_expected)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        action_pred = self.actor(states)
        actor_loss = -self.critic(states, action_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------------- soft update weights ------------------------ #
        TAU = 1e-3
        soft_update(self.actor, self.actor_target, TAU)
        soft_update(self.critic, self.critic_target, TAU)

    def save_model(self, folder_path, file_appendix = ''):
        torch.save(self.actor.state_dict(), folder_path + self.default_actor_weight_prefix + file_appendix + '.pth')
        torch.save(self.actor_target.state_dict(), folder_path + self.default_actor_target_weight_prefix + file_appendix + '.pth')
        torch.save(self.critic.state_dict(), folder_path + self.default_critic_weight_prefix + file_appendix + '.pth')
        torch.save(self.critic_target.state_dict(), folder_path + self.default_critic_target_weight_prefix + file_appendix + '.pth')

    def load_model(self, folder_path, file_appendix = ''):
        self.actor.load_state_dict(torch.load(folder_path + self.default_actor_weight_prefix + file_appendix + '.pth'))
        self.actor_target.load_state_dict(torch.load(folder_path + self.default_actor_target_weight_prefix + file_appendix + '.pth'))
        self.critic.load_state_dict(torch.load(folder_path + self.default_critic_weight_prefix + file_appendix + '.pth'))
        self.critic_target.load_state_dict(torch.load(folder_path + self.default_critic_target_weight_prefix + file_appendix + '.pth'))

# @dataclass
# class MultiAgentDDPGConfig:
#     num_state: int = 0
#     num_action: int = 0
#     num_agent: int = 1
#     actor_learning_rate: float = 1e-4
#     critic_learning_rate: float = 1e-4
#     batch_size: int = 128
#     update_every_timestamp: int = 20
#     update_time_each_stamp: int = 10
#     discount_factor: float = 0.99
#     replay_buffer_size: int = 1e6
#     weight_decay: float = 0

class MultiAgentDDPG():
    def __init__(self, config):
        self.config = config

        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")

        # declare actors. One actor per agent
        self.actors = [model.Actor(config.num_state, config.num_action).to(self.device) for _ in range(config.num_agent)]
        self.target_actors = [model.Actor(config.num_state, config.num_action).to(self.device) for _ in range(config.num_agent)]
        for actor, target_actor in zip(self.actors, self.target_actors):
            # initliaze target with the same parameters as local network
            target_actor.load_state_dict(actor.state_dict())
        self.actor_optimizers = [optim.Adam(actor.parameters(), lr=config.actor_learning_rate) for actor in self.actors]
        
        self.critics = [model.MADDPGCritic(config.num_state, config.num_action, config.num_agent).to(self.device) for _ in range(config.num_agent)]
        self.target_critics = [model.MADDPGCritic(config.num_state, config.num_action, config.num_agent).to(self.device) for _ in range(config.num_agent)]
        for critic, target_critic in zip(self.critics, self.target_critics):
            # initliaze target with the same parameters as local network
            target_critic.load_state_dict(critic.state_dict())
        self.critic_optimizers = [optim.Adam(critic.parameters(), lr=config.critic_learning_rate) for critic in self.critics]
        
        # initalize other variables
        self.t_step = 0
        self.memory = ReplayBufferMultiAgent(int(config.replay_buffer_size), config.batch_size, self.device)
        self.noise = OUNoise((config.num_agent, config.num_action), 2)
 
    def reset(self):
        self.noise.reset()

    def act(self, states):
        states = torch.from_numpy(states).float().to(self.device)
        final_actions = []
        for i in range(len(self.actors)):
            self.actors[i].eval()
            actions = self.actors[i](states[i]).cpu().data.numpy()
            final_actions.append(actions)
            self.actors[i].train()
        final_actions = np.array(final_actions)

        # apply noise and return
        final_actions += self.noise.sample()
        return np.clip(final_actions, -1., 1.)


    def step(self, state, action, reward, next_state, done):
        # update the memory first
        # all agents at the same timestamp are stored as a single experience
        # to form a joint experience
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) < self.config.batch_size:
            return 0., 0., False
        self.t_step = (self.t_step + 1) % self.config.update_every_timestamp
        if 0 != self.t_step:
            return 0., 0., False

        actor_losses = []
        critic_losses = []
        for _ in range(self.config.update_time_each_stamp):
            experiences = self.memory.sample()
            curr_actor_loss, curr_critic_loss = self.learn(experiences, self.config.discount_factor)
            actor_losses.append(curr_actor_loss)
            critic_losses.append(curr_critic_loss)
        actor_losses = np.array(actor_losses).reshape(1, -1)
        critic_losses = np.array(critic_losses).reshape(1, -1)
        return actor_losses.mean(), critic_losses.mean(), True

    def learn(self, experiences, discount_factor):
        states, actions, rewards, next_states, dones = experiences
        # Expected shapes:
        # states, next_states: [batch_size, num_agents, state_dim]
        # actions: [batch_size, num_agents, action_dim]
        # rewards, dones: [batch_size, num_agents]
        # assert self.config.num_agent == states.shape[1]
        # assert self.config.num_agent == next_states.shape[1]
        # assert self.config.num_agent == actions.shape[1]
        # assert self.config.num_agent == rewards.shape[1]
        # assert self.config.num_agent == dones.shape[1]

        batch_size = states.shape[0]

        states = states.view(batch_size, -1)
        next_states = next_states.view(batch_size, -1)
        actions = actions.view(batch_size, -1)
        rewards = rewards.view(batch_size, -1)
        dones = dones.view(batch_size, -1)

        all_actor_loss = []
        all_critic_loss = []
        for agent_idx in range(self.config.num_agent):
            # ---------------------------- update critic ---------------------------- #
            # use target actor to generate next actions for all agents
            flat_target_next_actions = []
            for i in range(self.config.num_agent):
                start_idx = i * self.config.num_state
                end_idx = (i + 1) * self.config.num_state
                target_next_actions = self.target_actors[i](next_states[:, start_idx : end_idx])
                flat_target_next_actions.append(target_next_actions)

            # dimenision: [batch_size, num_agent * action_size]
            flat_target_next_actions = torch.cat(flat_target_next_actions, dim=1)

            target_Q = self.target_critics[agent_idx](next_states, flat_target_next_actions)
            curr_reward = rewards[:, agent_idx].unsqueeze(1)
            curr_done = dones[:, agent_idx].unsqueeze(1)
            y = curr_reward + discount_factor * target_Q * (1. - curr_done).detach()

            q_expected = self.critics[agent_idx](states, actions)
            critic_loss = F.mse_loss(q_expected, y)
            all_critic_loss.append(np.abs(critic_loss.cpu().data.numpy()))
            
            self.critic_optimizers[agent_idx].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_idx].step()

            # ---------------------------- update actor ---------------------------- #
            # For updating the actor, compute the current actions for all agents.
            # Replace the action for agent_idx with the output from its actor network,
            # while using the other agents’ actions as computed by their current actors (detached).
            flat_all_actions = []
            for i in range(self.config.num_agent):
                start_idx = i * self.config.num_state
                end_idx = (i + 1) * self.config.num_state
                curr_action = self.actors[i](states[:, start_idx : end_idx])
                if i != agent_idx:
                    curr_action = curr_action.detach()
                flat_all_actions.append(curr_action)
            flat_all_actions = torch.cat(flat_all_actions, dim=1)
            actor_loss = -self.critics[agent_idx](states, flat_all_actions).mean()
            all_actor_loss.append(actor_loss.cpu().data.numpy())

            self.actor_optimizers[agent_idx].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_idx].step()

            # ------------------------- soft update weights ------------------------ #
            TAU = 1e-2
            soft_update(self.actors[agent_idx], self.target_actors[agent_idx], TAU)
            soft_update(self.critics[agent_idx], self.target_critics[agent_idx], TAU)

        all_actor_loss = np.array(all_actor_loss).reshape(1, -1)
        all_critic_loss = np.array(all_critic_loss).reshape(1, -1)
        return all_actor_loss.mean(), all_critic_loss.mean()

@dataclass
class MultiAgentPPOConfig:
    num_state: int = 0
    num_action: int = 0
    num_agent: int = 1
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    batch_size: int = 5
    clip_eps: float = 0.01
    entropy_coef: float = 0.1
    gamma: float = 0.99
    lam: float = 0.95
    replay_buffer_size: int = 1e6
    weight_decay: float = 0
    ppo_epochs: int = 10


# MAPPO Agent
class MAPPO:
    def __init__(self, obs_dim, act_dim, state_dim, num_agents,
                 gamma=0.99, clip_param=0.2, ppo_epochs=10, lr=3e-4,
                 batch_size=64, gae_lambda=0.95, entropy_coef=0.01):
        
        self.actors = nn.ModuleList([model.MAPPOActor(obs_dim, act_dim) for _ in range(num_agents)])
        self.critic = model.MAPPOCritic(state_dim)
        self.optimizer = optim.Adam([
            {'params': self.actors.parameters()},
            {'params': self.critic.parameters()}
        ], lr=lr)
        
        self.gamma = gamma
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.num_agents = num_agents
        self.act_dim = act_dim
        self.lr = lr
        # Add model directory parameter
        self.model_dir = "models/"
        os.makedirs(self.model_dir, exist_ok=True)

    def act(self, obs, agent_idx):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs)
            mu, std = self.actors[agent_idx](obs_tensor)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action = np.clip(action, -1., 1.)
            log_prob = dist.log_prob(action).sum(-1)
        return action.numpy(), log_prob.numpy()

    def update(self, buffer, curr_episode, max_episode):
        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(buffer.global_states))
        next_states = torch.FloatTensor(np.array(buffer.next_global_states))
        individual_obs = torch.FloatTensor(np.array(buffer.individual_obs))
        actions = torch.FloatTensor(np.array(buffer.actions))
        rewards = torch.FloatTensor(np.array(buffer.rewards))
        dones = torch.FloatTensor(np.array(buffer.dones))
        old_log_probs = torch.FloatTensor(np.array(buffer.log_probs))
        agent_indices = torch.LongTensor(np.array(buffer.agent_indices))

        # Calculate advantages using GAE
        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            
            advantages = torch.zeros_like(rewards)
            last_advantage = 0
            for t in reversed(range(len(rewards))):
                mask = 1.0 - dones[t]
                delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
                advantages[t] = delta + self.gamma * self.gae_lambda * mask * last_advantage
                last_advantage = advantages[t]
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Create dataset
        dataset = TensorDataset(individual_obs, actions, old_log_probs, 
                               advantages, returns, states, agent_indices)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # PPO update
        # the 1st 10k episodes do not decay
        decay_delay = 15000.
        learn_decay_ratio = max(0., curr_episode - decay_delay) / max_episode
        curr_clip_eps = self.clip_param #max(0.1, self.clip_param * (1. - learn_decay_ratio))
        curr_entropy_coef = self.entropy_coef # max(0.01, self.entropy_coef * (1. - learn_decay_ratio))
        curr_epochs = int(max(3, int(self.ppo_epochs * (1. - learn_decay_ratio))))
        final_lr = 1e-5  # More stability later
        self.optimizer.param_groups[0]["lr"] = max(final_lr, self.lr * (1. - learn_decay_ratio))
        for _ in range(curr_epochs):
            for batch in dataloader:
                ind_obs_b, actions_b, old_log_probs_b, adv_b, ret_b, states_b, agent_idx_b = batch
                
                policy_losses = []
                entropy_losses = []
                
                # Update each agent's policy
                for agent_id in range(self.num_agents):
                    mask = agent_idx_b == agent_id
                    if mask.sum() == 0:
                        continue
                    
                    # Agent-specific data
                    agent_obs = ind_obs_b[mask]
                    agent_actions = actions_b[mask]
                    agent_old_log_probs = old_log_probs_b[mask]
                    agent_adv = adv_b[mask]
                    
                    # Calculate new policy
                    mu, std = self.actors[agent_id](agent_obs)
                    dist = torch.distributions.Normal(mu, std)
                    new_log_probs = dist.log_prob(agent_actions).sum(-1)
                    entropy = dist.entropy().mean()
                    
                    # Policy loss
                    ratio = (new_log_probs - agent_old_log_probs).exp()
                    surr1 = ratio * agent_adv
                    surr2 = torch.clamp(ratio, 1.0 - curr_clip_eps, 
                                      1.0 + curr_clip_eps) * agent_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    policy_losses.append(policy_loss)
                    entropy_losses.append(entropy)

                # Combine losses
                policy_loss = torch.stack(policy_losses).mean() if policy_losses else 0.0
                entropy_loss = torch.stack(entropy_losses).mean() if entropy_losses else 0.0
                
                # Value loss
                values_pred = self.critic(states_b).squeeze()
                value_loss = F.mse_loss(values_pred, ret_b)
                
                # Total loss
                total_loss = policy_loss + value_loss - curr_entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer.step()
        
        buffer.clear()

    def save(self, episode=None, save_optimizer=True):
        """Save model checkpoints"""
        checkpoint = {
            # Save all actors' state dicts
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict() if save_optimizer else None,
            'num_agents': self.num_agents,  # Critical for verification
            'episode': episode
        }
        
        filename = f"mappo_checkpoint"
        if episode is not None:
            filename += f"_ep{episode}"
        filename += ".pth"
        
        torch.save(checkpoint, os.path.join(self.model_dir, filename))
        print(f"Saved checkpoint to {filename}")

    def load(self, path, load_optimizer=True):
        """Load model checkpoints"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
            
        checkpoint = torch.load(path)

        # Verify compatibility
        if checkpoint['num_agents'] != self.num_agents:
            raise ValueError(f"Checkpoint has {checkpoint['num_agents']} agents, "
                             f"but current setup has {self.num_agents}")
        
        # Load each actor individually
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(checkpoint['actors'][i])
        
        # Load optimizer if requested
        if load_optimizer and checkpoint['optimizer'] is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('episode', 0)

