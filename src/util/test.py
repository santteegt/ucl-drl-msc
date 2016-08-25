import gym
import numpy as np
import sys
import os

def load_action_set(filename, i, action_shape):
    actions = np.zeros((i, action_shape))
    row = 0
    with open(filename, mode='r') as f:
        for line in f:
            actions[row, ...] = line.split(',')[1:]
            row += 1
    return actions


def test(wolpertinger=False, k_prop=1):
    assert k_prop > 0, "Proportion for nearest neighbors must be non zero"
    env = gym.make('CollaborativeFiltering-v0')
    # env = gym.make('CartPole-v0')
    # env.monitor.start('/tmp/cf-1')
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    n_actions = 3883
    policy = env.action_space.sample
    import wolpertinger as wp
    if wolpertinger:
        action_set = load_action_set("data/embeddings-movielens1m.csv", n_actions, 119)
        policy = wp.Wolpertinger(env, i=n_actions, nn_index_file="indexStorageTest", action_set=action_set).g

    k = round(n_actions * k_prop) if k_prop < 1 else k_prop

    R = []
    for i_episode in range(2000):
        rew = 0.
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            # action = env.action_space.sample()

            action = policy(env.action_space.sample(), k=int(k)) if wolpertinger else policy()
            action = action[0][0] if wolpertinger else action
            observation, reward, done, info = env.step(action)
            rew += reward

            # print(info)
            if done:
                R.append(rew)
                # print("Episode finished after {} timesteps. Average Reward {}".format(t+1, np.mean(R)))
                with open("events.log", "a") as log:
                    log.write("Episode finished after {} timesteps. Average Reward {}\n".format(t + 1, np.mean(R)))
                break
        # print "Episode {} Average Reward per user: {}".format(i_episode, rew)
        with open("events.log", "a") as log:
            log.write("Episode {} Average Reward per user: {}\n".format(i_episode, rew))
        avr = np.mean(R)
        if env.spec.reward_threshold is not None and avr > env.spec.reward_threshold:
            # print "Threshold reached {} > {}".format(avr, env.spec.reward_threshold)
            with open("events.log", "a") as log:
                log.write("Threshold reached {} > {}\n".format(avr, env.spec.reward_threshold))
            break

    env.monitor.close()

if __name__ == "__main__":
    sys.path.append(os.path.dirname(__file__) + "/../")
    test(wolpertinger=True, k_prop=0.05)
