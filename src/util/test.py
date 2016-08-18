import gym
import numpy as np
def test():
    env = gym.make('CollaborativeFiltering-v0')
    # env = gym.make('CartPole-v0')
    # env.monitor.start('/tmp/cf-1')
    # env.reset()
    # for _ in range(1000):
    #     env.render()
    #     env.step(env.action_space.sample()) # take a random action
    R = []
    for i_episode in range(2000):
        rew = 0.
        observation = env.reset()
        for t in range(100):
            env.render()
            # print(observation)
            action = env.action_space.sample()
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
    test()
