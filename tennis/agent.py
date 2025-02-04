import model
from utils import OUNoise, ReplayBufferMultiAgent, ReplayBuffer, soft_update
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim

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


@dataclass
class MultiAgentDDPGConfig:
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
        
        self.critics = [model.MultiAgentCritic(config.num_state, config.num_action, config.num_agent).to(self.device) for _ in range(config.num_agent)]
        self.target_critics = [model.MultiAgentCritic(config.num_state, config.num_action, config.num_agent).to(self.device) for _ in range(config.num_agent)]
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
