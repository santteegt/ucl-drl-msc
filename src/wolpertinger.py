import numpy as np
import pyflann as fl
import gym
import filter_env
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('create_index', False, 'create the nearest-neighbors index from skratch')
flags.DEFINE_string('nn_index_file', None, 'Nearest neighbors index file')
flags.DEFINE_boolean('save_index', True, 'save the nearest-neighbors index on nn_index_file')
flags.DEFINE_float('nn_target_precision', 0.7, 'Nearest neighbors target precision for autotuned index')
flags.DEFINE_integer('knn', 1, 'Number of Nearest neighbors to lookup')
flags.DEFINE_integer('knn_checks', 2, 'Number of checks: N. of times the index tree should be recursively searched.' +
                     'It must be set when loading index from file')


class Wolpertinger(object):

    def __init__(self, env, i=1, nn_index_file=None):
        FLAGS.nn_index_file = nn_index_file if nn_index_file is not None else FLAGS.nn_index_file
        # FLAGS.create_index = create_index if create_index is not None else FLAGS.create_index
        self.__env = env
        self.__action_space = env.action_space
        self.__flann_params = None

        shape = self.__action_space.shape[0]
        lower = self.__action_space.low
        higher = self.__action_space.high
        i = i+1 if i == 1 else i
        self.__A = np.zeros((i, shape), dtype=lower.dtype)
        for d in np.arange(shape):
            low = lower[d]
            high = higher[d]
            self.__A[..., d] = np.linspace(low, high, dtype=low.dtype, num=i)

        self.__flann = fl.FLANN()
        fl.set_distance_type('euclidean')  # L2 norm distance

        if FLAGS.create_index:

            self.__flann_params = self.__flann.build_index(self.__A, algorithm="autotuned",
                                              target_precision=FLAGS.nn_target_precision, log_level="info")

            print('KNN index created with auto-tuned configuration params: {}.'.format(str(self.__flann_params)))

            if FLAGS.nn_index_file is not None:
                self.__flann.save_index(FLAGS.nn_index_file)
                print('KNN index file stored in {}'.format(FLAGS.nn_index_file))

        elif FLAGS.nn_index_file is not None:
            self.__flann.load_index(FLAGS.nn_index_file, self.__A)
        else:
            raise Exception("Error in parameter configuration. Index was not created/loaded")

    def g(self, action, k=None):
        '''
        Function g that returns the k-nearest-neighbors for a given continuous action
        Args:
            action: continuous action

        Returns:
            k-nearest-neighbors
        '''

        nearest_neighbors = k if k is not None else FLAGS.knn
        checks = self.__flann_params["checks"] if self.__flann_params else FLAGS.knn_checks
        results, dists = self.__flann.nn_index(action, nearest_neighbors, checks=checks)
        return [self.__A[val] for val in results]

    @property
    def discrete_actions(self):
        return self.__A




if __name__ == '__main__':
    env = filter_env.makeFilteredEnv(gym.make("InvertedDoublePendulum-v1"))
    x = Wolpertinger(env, i=1000000, nn_index_file="indexStorageTest")

    obs = env.reset()
    cont_action = env.action_space.sample()
    print('==Action in continuous space: {}'.format(cont_action))
    result = x.g(cont_action)
    print(result)