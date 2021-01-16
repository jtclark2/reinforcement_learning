import gym
import time

env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
observation = env.reset()
tic = time.time()
action = 0
total_reward = 0
for _ in range(1000):
    time.sleep(.01)
    env.render() # remove to speed up a LOT
    # observation, reward, done, info = env.step(env.action_space.sample()) # take a random action
    observation, reward, done, info = env.step(action) # take a random action
    total_reward += 1
    if(done != 0):
        print("Total Reward: %d" % (total_reward))
        print("DONE!")
        break
    pos, vel = observation
    if vel > 0:
        action = 2
    else:
        action = 0
toc = time.time()
print("Observation: ", observation)
# for env_name in gym.envs.registry.all():
#     print(env_name)
print("Simulation took %f seconds." % (toc-tic))
time.sleep(3)
env.close()
