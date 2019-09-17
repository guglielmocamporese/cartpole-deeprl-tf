##############################
# Imports
##############################

import gym
from classes import Agent, Episode
import numpy as np
import parameters as par


##############################
# Main
##############################

if __name__ == '__main__':

	# Create environment and agent
	env = gym.make('CartPole-v0')
	env._max_episode_steps = 500
	agent = Agent(env.action_space.n, env.observation_space.shape[0], name='agent')
	if par.TRAINED_AGENT_PATH != '':
		agent.load(par.TRAINED_AGENT_PATH)

	# Loop over episodes
	for e in range(par.NUM_EPISODES // par.EVERY_EPISODE):
		observations, returns, actions = [], [], []
		for _ in range(par.EVERY_EPISODE if par.TRAIN else 1):
			episode = Episode(env, render=False if par.TRAIN else True)
			o, r, a = episode.run(agent)
			if np.sum(episode.rewards) >= par.MIN_RETURN:
				observations += o; returns += r; actions += a;

		# Train
		if par.TRAIN:
			if len(observations) > 0:
				loss = agent.fit(observations, returns, actions)
				print('Episode: {:3d}, Return: {:.2f}, Loss: {:.3f}'.format((e + 1) * par.EVERY_EPISODE, np.sum(episode.rewards), loss))
		else:
			print('Episode: {:3d}, Return: {:.2f}'.format((e + 1), np.sum(episode.rewards)))

		# Save agent
		agent.save('./tmp')