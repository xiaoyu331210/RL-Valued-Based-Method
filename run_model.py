from unityagents import UnityEnvironment
import numpy as np
import agent
import model

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Declare agent
GAMMA = 0.99  # discount factor

agent = agent.Agent(brain.vector_observation_space_size, brain.vector_action_space_size, GAMMA)

# Function to train the network
from collections import deque

def train(agent, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):

    all_rewards = []
    rewards_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    for i_episode in range(n_episodes):
#         print("episode start")
        env_info = env.reset(train_mode=True)[brain_name]
#         print("got env")
        state = env_info.vector_observations[0]
        print(state)
        total_reward = 0
        eps = max(eps_end, eps * eps_decay)
        for t in range(max_t):
#             print("start t")
            # get action from the agent based on the curernt states
            action = agent.act(state, eps)
#             print("got action")
            # update the env based on the action
            env_info = env.step(action)[brain_name]
#             print("env step one")
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            # update the agent
            agent.step(state, action, reward, next_state, done)
#             print("agent step done")
            # update for the next iteration
            state = next_state
            total_reward += reward
            # the episode reachs the end, so need to start a new episode
            if done:
                break
        all_rewards.append(total_reward)
        rewards_window.append(total_reward)
#         print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(total_reward)), end="")
        print('Episode {}\tAverage Score: {:.2f}\n'.format(i_episode, np.mean(total_reward)), end="")
        if i_episode % 100 == 0:
#             print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('Episode {}\tAverage Score: {:.2f}\t Total Score: {:.2f}'.format(i_episode, np.mean(scores_window), np.mean(total_reward)))
        if np.mean(rewards_window)>=20.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            break
    
    return all_rewards
            

scores = train(agent, env, 2000, 10)
    




