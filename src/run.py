#!/usr/bin/env python
import os
import experiment
import gym
import numpy as np
import filter_env
import ddpg
import wolpertinger as wp
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('upload',False,'upload to gym (requires evironment variable OPENAI_GYM_API_KEY)')
flags.DEFINE_string('env','','gym environment')
# flags.DEFINE_integer('train',10000,'training time between tests. use 0 for test run only')
# flags.DEFINE_integer('test',10000,'testing time between training')
flags.DEFINE_integer('train',10,'training time between tests. use 0 for test run only')
flags.DEFINE_integer('test',1,'testing time between training')
flags.DEFINE_integer('tmax',10000,'maximum timesteps per episode')
flags.DEFINE_bool('random',False,'use random agent')
flags.DEFINE_bool('tot',False,'train on test data')
flags.DEFINE_integer('total',1000000,'total training time')
flags.DEFINE_float('monitor',.05,'probability of monitoring a test episode')
flags.DEFINE_bool('wolpertinger',False,'Train critic using the full Wolpertinger policy')
flags.DEFINE_integer('wp_total_actions', 1000000, 'total number of actions to discretize under the Wolpertinger policy')
# ...
# TODO: make command line options

VERSION = 'DDPG-v0'
GYM_ALGO_ID = 'alg_TmtzkcfSauZoBF97o9aQ'

if FLAGS.random:
  FLAGS.train = 0

class Experiment:
  def run(self):
    self.t_train = 0
    self.t_test = 0

    # create filtered environment
    self.env = filter_env.makeFilteredEnv(gym.make(FLAGS.env))
    # self.env = gym.make(FLAGS.env)
    
    self.env.monitor.start(FLAGS.outdir+'/monitor/',video_callable=lambda _: False)
    # self.env.monitor.start(FLAGS.outdir+'/monitor/',video_callable=lambda _: True)
    gym.logger.setLevel(gym.logging.WARNING)

    dimO = self.env.observation_space.shape
    dimA = self.env.action_space.shape
    print(dimO,dimA)

    import pprint
    pprint.pprint(self.env.spec.__dict__,width=1)

    wolp = None
    if FLAGS.wolpertinger:
      wolp = wp.Wolpertinger(self.env, i=FLAGS.wp_total_actions).g

    self.agent = ddpg.Agent(dimO=dimO, dimA=dimA, custom_policy=FLAGS.wolpertinger,
                            env_dtype=str(self.env.action_space.high.dtype))

    returns = []

    # main loop
    while self.t_train < FLAGS.total:

      # test
      T = self.t_test
      R = []
      while self.t_test - T < FLAGS.test:
        R.append(self.run_episode(test=True, monitor=(self.t_test - T < FLAGS.monitor * FLAGS.test), custom_policy=wolp))
        self.t_test += 1
      avr = np.mean(R)
      print('Average test return\t{} after {} timesteps of training'.format(avr,self.t_train))
      # save return
      returns.append((self.t_train, avr))
      np.save(FLAGS.outdir+"/returns.npy",returns)

      # evaluate required number of episodes for gym and end training when above threshold
      if self.env.spec.reward_threshold is not None and avr > self.env.spec.reward_threshold:
        avr = np.mean([self.run_episode(test=True) for _ in range(self.env.spec.trials)]) # trials???
        if avr > self.env.spec.reward_threshold:
          break

      # train
      T = self.t_train
      R = []
      while self.t_train - T < FLAGS.train:
        R.append(self.run_episode(test=False, custom_policy=wolp))
        self.t_train += 1
      avr = np.mean(R)
      print('Average training return\t{} after {} timesteps of training'.format(avr,self.t_train))

    self.env.monitor.close()
    # upload results
    if FLAGS.upload:
      gym.upload(FLAGS.outdir+"/monitor",algorithm_id = GYM_ALGO_ID)

  def run_episode(self,test=True,monitor=False, custom_policy=None):
    self.env.monitor.configure(lambda _: test and monitor)
    observation = self.env.reset()
    self.agent.reset(observation)
    R = 0. # return
    t = 1
    term = False
    while not term:
      # self.env.render(mode='human')

      if FLAGS.random:
        action = self.env.action_space.sample()
      else:
        action = self.agent.act(test=test)

      # Run Wolpertinger discretization
      action_to_perform = action
      g_action = None
      if FLAGS.wolpertinger:
        A_k = custom_policy(action)
        rew_g = R * np.ones(len(A_k), dtype=self.env.action_space.high.dtype)
        term_g = np.zeros(len(A_k), dtype=np.bool)
        # for i in range(len(A_k)):
        #   _, rew_g[i], term_g[i], _ = self.env.step(A_k[i])
        maxq_index = self.agent.wolpertinger_policy(action, A_k, rew_g, term_g)
        g_action = A_k[maxq_index[0]]
        action_to_perform = g_action
        # res = self.agent.wolpertinger_policy(action, A_k, rew_g, term_g)
        # print('continuous action: {} discretized action: {}'.format(action, g_action))


      # observation, reward, term, info = self.env.step(action)
      observation, reward, term, info = self.env.step(action_to_perform)
      term = (t >= FLAGS.tmax) or term

      r_f = self.env.filter_reward(reward)
      self.agent.observe(r_f,term,observation,test = test and not FLAGS.tot, g_action=g_action)

      # if test:
      #   self.t_test += 1
      # else:
      #   self.t_train += 1

      R += reward
      t += 1


    return R


def main():
  Experiment().run()

if __name__ == '__main__':
  os.environ['PATH'] += ':/usr/local/bin' # to be able to run under gym ffmpeg dependency under IDE
  experiment.run(main)
