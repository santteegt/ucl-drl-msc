import numpy as np
import gym
import filter_env
import tensorflow as tf
import graphlab as gl
import time


flags = tf.app.flags
FLAGS = flags.FLAGS
# flags.DEFINE_integer('knn', 1, 'Number of Nearest neighbors to lookup')


class FMPolicy(object):

    def __init__(self, env, create_model=False, data_set=None):
        self._env = env
        self._model = None
        if not create_model:
            self._model = gl.load_model('data/FMmodel')

        print "FM policy model loaded."

    def g(self, state, action, k=None):
        '''
        Function g that returns the k-nearest-neighbors for a given continuous action
        Args:
            state: current state
            action: continuous action

        Returns:
            k-nearest-neighbors
        '''

        # assert type(state) is int and len(action) == 1, \
        #     "This policy only accepts state and type action spaces as int and single values"
        action = np.clip(np.round(action), self._env.action_space.low, self._env.action_space.high)
        # action as integer id
        action = int("".join([str(int(a)) for a in action]), 2)
        vector_length = len(self._env.action_space.high)
        if action > 0:
            timestamp = long(time.time() * 1000) - 874724710
            observed_items = gl.SFrame({'item_id': [action], 'timestamp': [timestamp], 'prev_item': [state]})

            nearest_neighbors = k if k is not None else FLAGS.knn
            k_interactions = self._model.recommend_from_interactions(observed_items, k=nearest_neighbors, diversity=1)

            items = ["{0:0{1}b}".format(item[0], vector_length) for item in k_interactions[["item_id"]].to_numpy()]
        else:
            items = ["00000000000"]

        item_vectors = []
        for item in items:
            item_vectors.append(np.array([float(item[i]) for i in range(vector_length)]))

        return item_vectors

if __name__ == '__main__':
    # env = filter_env.makeFilteredEnv(gym.make("InvertedDoublePendulum-v1"))
    env = filter_env.makeFilteredEnv(gym.make("CollaborativeFiltering-v3"))
    x = FMPolicy(env)

    obs = env.reset()
    cont_action = env.action_space.sample()
    print('==Action in continuous space: {}'.format(cont_action))
    result = x.g(cont_action)
    print(result)