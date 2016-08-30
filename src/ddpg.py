import tensorflow as tf
import ddpg_nets_dm as nets_dm
from replay_memory import ReplayMemory
import numpy as np
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('ou_sigma',0.2,'')
# flags.DEFINE_integer('warmup',50000,'time without training but only filling the replay memory')
flags.DEFINE_integer('warmup',5000,'time without training but only filling the replay memory')
flags.DEFINE_bool('warmq',True,'train Q during warmup time')
flags.DEFINE_float('log',.01,'probability of writing a tensorflow log at each timestep')
flags.DEFINE_integer('bsize',32,'minibatch size')
flags.DEFINE_bool('async',True,'update policy and q function concurrently')
flags.DEFINE_bool('batch_norm',False,'execute batch normalization over network layers')
flags.DEFINE_integer('iter', 5, 'train iterations each timestep')

# ...
# TODO: make command line options
tau =.001
discount =.99
pl2 =.0
ql2 =.01
lrp =.0001
lrq =.001
# ql2 =.001
# lrp =.0015
# lrq =.005
ou_theta = 0.15
ou_sigma = 0.2
rm_size = 500000
# rm_dtype = 'float32'
threads = 8


class Agent:
  """
  DDPG Agent
  """

  started_train = False

  def __init__(self, dimO, dimA, custom_policy=False, env_dtype=tf.float32):
    dimA = list(dimA)
    dimO = list(dimO)

    nets=nets_dm

    self.custom_policy = custom_policy

    # init replay memory
    self.rm = ReplayMemory(rm_size, dimO, dimA, dtype=np.__dict__[env_dtype])
    # start tf session
    self.sess = tf.Session(config=tf.ConfigProto(
      inter_op_parallelism_threads=threads,
      log_device_placement=True,
      allow_soft_placement=True))

    # create tf computational graph
    #
    self.theta_p = nets.theta_p(dimO, dimA)
    self.theta_q = nets.theta_q(dimO, dimA)
    self.theta_pt, update_pt = exponential_moving_averages(self.theta_p, tau)
    self.theta_qt, update_qt = exponential_moving_averages(self.theta_q, tau)

    obs = tf.placeholder(env_dtype, [None] + dimO, "obs")
    is_training = tf.placeholder(tf.bool, name="is_training")
    # act_test, sum_p = nets.policy(obs, self.theta_p)
    act_test, sum_p = nets.policy(obs, self.theta_p) if not FLAGS.batch_norm else nets.policy_norm(obs, self.theta_p, is_training)

    # explore
    noise_init = tf.zeros([1]+dimA, dtype=env_dtype)
    noise_var = tf.Variable(noise_init)
    self.ou_reset = noise_var.assign(noise_init)
    noise = noise_var.assign_sub((ou_theta) * noise_var - tf.random_normal(dimA, stddev=ou_sigma, dtype=env_dtype))
    act_expl = act_test + noise

    # for Wolpertinger full policy
    act_cont = tf.placeholder(env_dtype, [None] + dimA, "action_cont_space")

    # g_actions = tf.placeholder(env_dtype, [FLAGS.knn] + dimA, "knn_actions")
    g_actions = tf.placeholder(env_dtype, [None] + dimA, "knn_actions")

    # rew_g = tf.placeholder(env_dtype, [FLAGS.knn] + dimA, "rew")

    # rew_g = tf.placeholder(env_dtype, [FLAGS.knn], "rew_g")
    # term_g = tf.placeholder(tf.bool, [FLAGS.knn], "term_g")
    rew_g = tf.placeholder(env_dtype, [1], "rew_g")
    term_g = tf.placeholder(tf.bool, [1], "term_g")

    # g_dot_f = tf.mul(g_actions, act_cont, "g_dot_f")
    g_dot_f = g_actions
    q_eval, _ = nets.qfunction(obs, g_dot_f, self.theta_q) if not FLAGS.batch_norm else nets.qfunction_norm(obs,
                                                                                                            g_dot_f,
                                                                                                            self.theta_q,
                                                                                                            is_training,
                                                                                                            reuse=True)
    # wolpertinger_policy = tf.stop_gradient( tf.argmax( tf.select(term_g, rew_g, rew_g + discount * q_eval),
    #                                                        dimension=0, name="q_max") )
    wolpertinger_policy = tf.stop_gradient(tf.select(term_g, rew_g, rew_g + discount * q_eval))

    # test
    # q, sum_q = nets.qfunction(obs, act_test, self.theta_q)
    q, sum_q = nets.qfunction(obs, act_test, self.theta_q) if not FLAGS.batch_norm else nets.qfunction_norm(obs, act_test, self.theta_q, is_training)
    # training
    # policy loss
    meanq = tf.reduce_mean(q, 0)
    wd_p = tf.add_n([pl2 * tf.nn.l2_loss(var) for var in self.theta_p])  # weight decay
    loss_p = -meanq + wd_p #???
    # policy optimization
    optim_p = tf.train.AdamOptimizer(learning_rate=lrp)
    grads_and_vars_p = optim_p.compute_gradients(loss_p, var_list=self.theta_p)
    optimize_p = optim_p.apply_gradients(grads_and_vars_p)
    with tf.control_dependencies([optimize_p]):
      train_p = tf.group(update_pt)

    # q optimization
    act_train = tf.placeholder(env_dtype, [FLAGS.bsize] + dimA, "act_train")
    g_act_train = tf.placeholder(env_dtype, [FLAGS.bsize] + dimA, "g_act_train")
    rew = tf.placeholder(env_dtype, [FLAGS.bsize], "rew")
    obs2 = tf.placeholder(env_dtype, [FLAGS.bsize] + dimO, "obs2")
    term2 = tf.placeholder(tf.bool, [FLAGS.bsize], "term2")

    # FOR WOLPERTINGER FUNCTIONALITY: eval wheter the agent is using pure DDPG or DDPG + Wolpertinger
    tensor_cond = tf.constant(self.custom_policy, dtype=tf.bool, name="is_custom_p")

    # full_act_policy = tf.cond(tensor_cond,
    #                           # lambda: tf.mul(g_act_train, act_train, name="full_act_policy"),
    #                           lambda: g_act_train,
    #                           lambda: act_train,
    #                           )

    # q
    # q_train, sum_qq = nets.qfunction(obs, act_train, self.theta_q)

    # TAKING THE POLICY GRADIENT AT THE ACTUAL OUTPUT OF f
    q_train, sum_qq = nets.qfunction(obs, act_train, self.theta_q) if not FLAGS.batch_norm else \
      nets.qfunction_norm(obs, act_train, self.theta_q, is_training, reuse=True)
    # q_train, sum_qq = nets.qfunction(obs, full_act_policy, self.theta_q) if not FLAGS.batch_norm else \
    #   nets.qfunction_norm(obs, full_act_policy, self.theta_q, is_training, reuse=True)

    # q targets
    # act2, sum_p2 = nets.policy(obs2, theta=self.theta_pt)
    act2, sum_p2 = nets.policy(obs2, theta=self.theta_pt) if not FLAGS.batch_norm else nets.policy_norm(obs2, theta=self.theta_pt, is_training=is_training, reuse=True)

    # WOLPERTINGER FUNCTIONALITY: The target action in the Q-update is generated by the full policy and not simply f
    full_act_policy2 = tf.cond(tensor_cond,
                              # lambda: tf.mul(g_act_train, act2, name="full_act_policy"),
                              lambda: g_act_train,
                              lambda: act2,
                              )
    # q2, sum_q2 = nets.qfunction(obs2, act2, theta=self.theta_qt)
    # q2, sum_q2 = nets.qfunction(obs2, act2, theta=self.theta_qt) if not FLAGS.batch_norm else nets.qfunction_norm(obs2, act2, theta=self.theta_qt, is_training=is_training, reuse=True)
    q2, sum_q2 = nets.qfunction(obs2, full_act_policy2, theta=self.theta_qt) if not FLAGS.batch_norm else nets.qfunction_norm(obs2,
                                                                                                                  full_act_policy2,
                                                                                                                  theta=self.theta_qt,
                                                                                                                  is_training=is_training,
                                                                                                                  reuse=True)
    q_target = tf.stop_gradient(tf.select(term2,rew,rew + discount*q2))
    # q_target = tf.stop_gradient(rew + discount * q2)
    # q loss
    td_error = q_train - q_target # TODO: maybe it needs to be q_target - q_train
    ms_td_error = tf.reduce_mean(tf.square(td_error), 0)
    wd_q = tf.add_n([ql2 * tf.nn.l2_loss(var) for var in self.theta_q])  # weight decay
    loss_q = ms_td_error + wd_q
    # q optimization
    optim_q = tf.train.AdamOptimizer(learning_rate=lrq)
    grads_and_vars_q = optim_q.compute_gradients(loss_q, var_list=self.theta_q)
    optimize_q = optim_q.apply_gradients(grads_and_vars_q)
    with tf.control_dependencies([optimize_q]):
      train_q = tf.group(update_qt)

    # logging
    log_obs = [] if dimO[0]>20 else [tf.histogram_summary("obs/"+str(i),obs[:,i]) for i in range(dimO[0])]
    log_act = [] if dimA[0]>20 else [tf.histogram_summary("act/inf"+str(i),act_test[:,i]) for i in range(dimA[0])]
    log_act2 = [] if dimA[0]>20 else [tf.histogram_summary("act/train"+str(i),act_train[:,i]) for i in range(dimA[0])]
    log_misc = [sum_p, sum_qq, tf.histogram_summary("td_error", td_error)]
    log_grad = [grad_histograms(grads_and_vars_p), grad_histograms(grads_and_vars_q)]
    log_train = log_obs + log_act + log_act2 + log_misc + log_grad

    # initialize tf log writer
    self.writer = tf.train.SummaryWriter(FLAGS.outdir+"/tf", self.sess.graph, flush_secs=20)

    # init replay memory for recording episodes
    max_ep_length = 10000
    self.rm_log = ReplayMemory(max_ep_length,dimO,dimA, env_dtype)

    # tf functions
    with self.sess.as_default():
      # self._act_test = Fun(obs,act_test)
      # self._act_expl = Fun(obs,act_expl)
      # self._reset = Fun([],self.ou_reset)
      # self._train_q = Fun([obs,act_train,rew,obs2,term2],[train_q],log_train,self.writer)
      # self._train_p = Fun([obs],[train_p],log_train,self.writer)
      # self._train = Fun([obs,act_train,rew,obs2,term2],[train_p,train_q],log_train,self.writer)

      self._act_test = Fun([obs, is_training],act_test)
      self._act_expl = Fun([obs, is_training],act_expl)
      self._reset = Fun([],self.ou_reset)
      self._train_q = Fun([obs, act_train, g_act_train, rew, obs2, term2, is_training], [train_q], log_train, self.writer)
      self._train_p = Fun([obs, is_training],[train_p],log_train,self.writer)
      self._train = Fun([obs, act_train, g_act_train, rew, obs2, term2, is_training], [train_p, train_q], log_train, self.writer)
      self._wolpertinger_p = Fun([obs, act_cont, g_actions, rew_g, term_g, is_training], [wolpertinger_policy])

    # initialize tf variables
    self.saver = tf.train.Saver(max_to_keep=1)
    ckpt = tf.train.latest_checkpoint(FLAGS.outdir+"/tf")
    if ckpt:
      self.saver.restore(self.sess,ckpt)
    else:
      self.sess.run(tf.initialize_all_variables())

    self.sess.graph.finalize()

    self.t = 0  # global training time (number of observations)

  def reset(self, obs):
    self._reset()
    self.observation = obs  # initial observation

  def act(self, test=False):
    obs = np.expand_dims(self.observation, axis=0)
    # action = self._act_test(obs) if test else self._act_expl(obs)
    action = self._act_test(obs, False) if test else self._act_expl(obs, True)
    self.action = np.atleast_1d(np.squeeze(action, axis=0)) # TODO: remove this hack
    return self.action

  def wolpertinger_policy(self, action_cont, g_actions, rew_g, term_g):
    obs = np.expand_dims(self.observation, axis=0)
    action_cont = np.expand_dims(action_cont, axis=0)
    # rew_g = np.expand_dims(rew_g, axis=0)
    # return np.asarray( self._wolpertinger_p(obs, action_cont, g_actions, rew_g, term_g) )
    i = 0
    q_values = []
    for g_action in g_actions:
      g_action = np.expand_dims(g_action, axis=0)
      q_values.append(self._wolpertinger_p(obs, action_cont, g_action, [rew_g[i]], [term_g[i]])[0])
      i += 1

    # return self._wolpertinger_p(obs, action_cont, g_actions, rew_g, term_g)[0]
    return np.argmax(q_values)

  def observe(self, rew, term, obs2, test=False, g_action=None):

    obs1 = self.observation
    self.observation = obs2

    # train
    if not test:
      self.t = self.t + 1
      self.rm.enqueue(obs1, term, self.action, g_action, rew)

      if self.t > FLAGS.warmup:
        self.train()

      elif FLAGS.warmq and self.rm.n > 1000:
        # Train Q on warmup
        obs, act, g_act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
        # self._train_q(obs,act,rew,ob2,term2, log = (np.random.rand() < FLAGS.log), global_step=self.t)
        for i in xrange(FLAGS.iter):
          self._train_q(obs, act, g_act, rew, ob2, term2, True, log = (np.random.rand() < FLAGS.log), global_step=self.t)

      # save parameters etc.
      # if (self.t+45000) % 50000 == 0: # TODO: correct
      #   s = self.saver.save(self.sess,FLAGS.outdir+"f/tf/c",self.t)
      #   print("DDPG Checkpoint: " + s)

  def train(self):
    if not self.started_train:
        with open(os.path.join(FLAGS.outdir, "output.log"), mode='a') as f:
          f.write('===> Warm up complete\n')
        self.started_train = True

    obs, act, g_act, rew, ob2, term2, info = self.rm.minibatch(size=FLAGS.bsize)
    log = (np.random.rand() < FLAGS.log)

    if FLAGS.async:
      # self._train(obs,act,rew,ob2,term2, log = log, global_step=self.t)
      for i in xrange(FLAGS.iter):
        self._train(obs, act, g_act, rew, ob2, term2, True, log = log, global_step=self.t)
    else:
      # self._train_q(obs,act,rew,ob2,term2, log = log, global_step=self.t)
      # self._train_p(obs, log = log, global_step=self.t)
      for i in xrange(FLAGS.iter):
        self._train_q(obs, act, g_act, rew, ob2, term2, True, log=log, global_step=self.t)
        self._train_p(obs, True, log = log, global_step=self.t)

  def write_scalar(self,tag,val):
    s = tf.Summary(value=[tf.Summary.Value(tag=tag,simple_value=val)])
    self.writer.add_summary(s,self.t)


  def __del__(self):
    self.sess.close()



# Tensorflow utils
#
class Fun:
  """ Creates a python function that maps between inputs and outputs in the computational graph. """
  def __init__(self, inputs, outputs,summary_ops=None,summary_writer=None, session=None ):
    self._inputs = inputs if type(inputs)==list else [inputs]
    self._outputs = outputs
    self._summary_op = tf.merge_summary(summary_ops) if type(summary_ops)==list else summary_ops
    self._session = session or tf.get_default_session()
    self._writer = summary_writer
  def __call__(self, *args, **kwargs):
    """
    Arguments:
      **kwargs: input values
      log: if True write summary_ops to summary_writer
      global_step: global_step for summary_writer
    """
    log = kwargs.get('log',False)

    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg

    out = self._outputs + [self._summary_op] if log else self._outputs
    res = self._session.run(out, feeds)
    
    if log:
      i = kwargs['global_step']
      self._writer.add_summary(res[-1],global_step=i)
      res = res[:-1]

    return res

def grad_histograms(grads_and_vars):
  s = []
  for grad, var in grads_and_vars:
    s.append(tf.histogram_summary(var.op.name + '', var))
    s.append(tf.histogram_summary(var.op.name + '/gradients', grad))
  return tf.merge_summary(s)

def exponential_moving_averages(theta, tau=0.001):
  ema = tf.train.ExponentialMovingAverage(decay=1 - tau)
  update = ema.apply(theta)  # also creates shadow vars
  averages = [ema.average(x) for x in theta]
  return averages, update
