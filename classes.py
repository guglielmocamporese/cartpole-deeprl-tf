##############################
# Imports
##############################

import tensorflow as tf
import parameters as par
import numpy as np
import os


##############################
# Agent class
##############################

class Agent(object):
	def __init__(self, num_actions, observation_space_size, name='agent'):
		self.num_actions = num_actions
		self.observation_space_size = observation_space_size
		self.name = name
		self.observations = tf.placeholder(tf.float32, [None, self.observation_space_size], name='observations')
		self.logits = self.get_logits(self.observations)
		self.probs = tf.nn.softmax(self.logits)
		self.returns = tf.placeholder(tf.float32, [None], name='returns')
		self.actions = tf.placeholder(tf.int32, [None], name='actions')
		self.saver = tf.train.Saver()
		self.loss, self.train_step, self.sess = self.compile()

	def get_logits(self, x):
		x = tf.layers.dense(x, 128, activation=tf.nn.relu)
		x = tf.layers.dense(x, self.num_actions)
		return x

	def sample(self, x):
		action = tf.multinomial(self.logits, 1)
		return self.sess.run(action, feed_dict={self.observations: x})[0, 0]

	def compile(self, lr=1e-2):
		optimizer = tf.train.AdamOptimizer(learning_rate=lr)
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(self.actions, self.num_actions), logits=self.logits)
		w_loss = tf.reduce_mean(loss * self.returns)
		train_step = optimizer.minimize(w_loss)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		return w_loss, train_step, sess

	def fit(self, obs, r, actions):
		_, loss = self.sess.run([self.train_step, self.loss], feed_dict={self.observations: obs, self.returns: r, self.actions: actions})
		return loss

	def save(self, path):
		if not os.path.exists(path):
			os.makedirs(path)
		self.saver.save(self.sess, os.path.join(path, self.name))

	def load(self, path):
		self.saver.restore(self.sess, path)
		print('Agent loaded: {}'.format(path))

	def __del__(self):
		self.sess.close()


##############################
# Episode class
##############################

class Episode(object):
	def __init__(self, env, render=False):
		self.env = env
		self.render = render
		self.actions = []
		self.rewards = []
		self.observations = []
		self.length = 0

	def get_env(self):
		return self.env

	def update(self, obs, reward, action):
		self.observations.append(obs)
		self.rewards.append(reward)
		self.actions.append(action)
		self.length += 1

	def run(self, agent):
		obs = self.env.reset()
		self.observations.append(obs)
		done = False
		while not done:
			if self.render:
				self.env.render()
			action = agent.sample(np.array([obs]))
			obs, reward, done, info = self.env.step(action)
			self.update(obs, reward, action)

			if done:
				self.observations.pop()
				self.compute_returns()
		return self.observations, self.returns, self.actions

	def compute_returns(self, gamma=1.0):
		returns = [0]
		for i in reversed(range(self.length)):
			r = self.rewards[i] + gamma * returns[0]
			returns.insert(0, r)
		returns.pop()
		self.returns = returns

	def __del__(self):
		self.env.close()
