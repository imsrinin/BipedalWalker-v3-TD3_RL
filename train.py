
import torch
import gym
import numpy as np
from TD3 import TD3
from utils import ReplayBuffer
from collections import deque
import pickle
import matplotlib.pyplot as plt
from typing import Union
from gym import spaces

######### Custom Gym Wrapper Function #########

class CustomWrapper(gym.Wrapper):
    # def __init__(self, env):
    #     super().__init__(env)

    #state space
    # Num	    Observation	                Min	    Max 	Mean
    # 0        	hull_angle	                0	    2*pi	0.5
    # 1     	hull_angularVelocity	    -inf	+inf	-
    # 2     	vel_x	                    -1	    +1	    -
    # 3     	vel_y	                    -1	    +1	    -
    # 4	        hip_joint_1_angle	        -inf	+inf	-
    # 5     	hip_joint_1_speed	        -inf	+inf	-
    # 6     	knee_joint_1_angle	        -inf	+inf	-
    # 7     	knee_joint_1_speed	        -inf	+inf	-
    # 8     	leg_1_ground_contact_flag	0	    1	    -
    # 9	        hip_joint_2_angle	        -inf	+inf	-
    # 10        hip_joint_2_speed	        -inf	+inf	-
    # 11        knee_joint_2_angle	        -inf	+inf	-
    # 12        knee_joint_2_speed	        -inf	+inf	-
    # 13        leg_2_ground_contact_flag	0	    1	    -
    # 14-23	10  lidar readings	            -inf	+inf	-

    #action space
    # Num	Name	                    Min	Max
    # 0 	Hip_1 (Torque / Velocity)	-1	+1
    # 1	    Knee_1 (Torque / Velocity)	-1	+1
    # 2	    Hip_2 (Torque / Velocity)	-1	+1
    # 3	    Knee_2 (Torque / Velocity)	-1	+1

    def __init__(
        self,
        env: gym.Env,
        min_action: Union[float, int, np.ndarray],
        max_action: Union[float, int, np.ndarray],
    ):
        """Initializes the :class:`RescaleAction` wrapper.
        Args:
            env (Env): The environment to apply the wrapper
            min_action (float, int or np.ndarray): The min values for each action. This may be a numpy array or a scalar.
            max_action (float, int or np.ndarray): The max values for each action. This may be a numpy array or a scalar.
        """
        assert isinstance(
            env.action_space, spaces.Box
        ), f"expected Box action space, got {type(env.action_space)}"
        assert np.less_equal(min_action, max_action).all(), (min_action, max_action)

        super().__init__(env)
        self.min_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + min_action
        )
        self.max_action = (
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + max_action
        )
        self.action_space = spaces.Box(
            low=min_action,
            high=max_action,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype,
        )

    def step(self, action):
        # modify obs
        obs, reward, terminated, info = self.env.step(action)
        obs = obs[:20]
        return obs, reward, terminated, info

    def reset(self):
        obs = self.env.reset()
        obs = obs[:20]
        return obs

    def action(self, action):
        """Rescales the action affinely from  [:attr:`min_action`, :attr:`max_action`] to the action space of the base environment, :attr:`env`.
        Args:
            action: The action to rescale
        Returns:
            The rescaled action
        """
        assert np.all(np.greater_equal(action, self.min_action)), (
            action,
            self.min_action,
        )
        assert np.all(np.less_equal(action, self.max_action)), (action, self.max_action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * (
            (action - self.min_action) / (self.max_action - self.min_action)
        )
        action = np.clip(action, low, high)
        return action

######### Hyperparameters #########
gym.logger.set_level(40)
env_name = "BipedalWalker-v3"
# env = CustomWrapper(gym.make(env_name),  min_action = -0.5,  max_action = 0.5)
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
# state_dim = 20
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print('max action', max_action)
# print('obs space ',env.observation_space.shape[0])
# print('action space high',env.action_space.high)
log_interval = 100  # print avg reward after interval
random_seed = 0
gamma = 0.99  # discount for future rewards
batch_size = 100  # num of transitions sampled from replay buffer
lr = 0.001
exploration_noise = 0.1
polyak = 0.995  # target policy update parameter (1-tau)
policy_noise = 0.2  # target policy smoothing noise
noise_clip = 0.3
policy_delay = 2  # delayed policy updates parameter
max_episodes = 10000  # max num of episodes
max_timesteps = 2000  # max timesteps in one episode
directory = "./preTrained/"  # save trained models
filename = "TD3_{}_{}".format(env_name, random_seed)

start_episode = 0

print(state_dim)
policy = TD3(lr, state_dim, action_dim, max_action)
replay_buffer = ReplayBuffer()

if random_seed:
    print("Random Seed: {}".format(random_seed))
    env.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

LOAD = False
if LOAD:
    start_episode = 6
    policy.load(directory, filename, str(start_episode))

# logging variables:
scores = []
mean_scores = []
last_scores = deque(maxlen=log_interval)
distances = []
mean_distances = []
last_distance = deque(maxlen=log_interval)
losses_mean_episode = []
prune_ct = 200
# training procedure:
for ep in range(start_episode + 1, max_episodes + 1):
    state = env.reset()
    total_reward = 0
    total_distance = 0
    actor_losses = []
    c1_losses = []
    c2_losses = []
    # if ep % prune_ct == 0:
    #     policy.prune_model()
    #     print('i have pruned your model by 10%')
    for t in range(max_timesteps):
        # select action and add exploration noise:

        action = policy.select_action(state)
        # print('action size',action.shape)
        action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
        action = action.clip(env.action_space.low, env.action_space.high)

        # take action in env:
        next_state, reward, done, _ = env.step(action)
        
        replay_buffer.add((state, action, reward, next_state, float(done)))
        state = next_state
        # print('state size', state.shape)
        total_reward += reward
        if reward != -100:
            total_distance += reward

        # if episode is done then update policy:
        if done or t == (max_timesteps - 1):
            actor_loss, c1_loss, c2_loss = policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
            actor_losses.append(actor_loss)
            c1_losses.append((c1_loss))
            c2_losses.append(c2_loss)
            break
    mean_loss_actor = np.mean(actor_losses)
    mean_loss_c1 = np.mean(c1_losses)
    mean_loss_c2 = np.mean(c2_losses)
    losses_mean_episode.append((ep, mean_loss_actor, mean_loss_c1, mean_loss_c2))
    print('\rEpisode: {}/{},\tScore: {:.2f},\tDistance: {:.2f},\tactor_loss: {},\tc1_loss:{},\tc2_loss:{}'.format(ep, max_episodes,total_reward,total_distance,mean_loss_actor,mean_loss_c1, mean_loss_c2))

    # logging updates:
    scores.append(total_reward)
    distances.append(total_distance)
    last_scores.append(total_reward)
    last_distance.append(total_distance)
    mean_score = np.mean(last_scores)
    mean_distance = np.mean(last_distance)
    FILE = 'record.dat'
    data = [ep, total_reward, total_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2]
    with open(FILE, "ab") as f:
        pickle.dump(data, f)

    # if avg reward > 300 then save and stop traning:
    if (mean_score) >= 300:
        print("########## Solved! ###########")
        name = filename + '_solved'
        policy.save(directory, name, str(ep))
        break

    # print avg reward every log interval:
    if ep % log_interval == 0:
        policy.save(directory, filename, str(ep))
        mean_scores.append(mean_score)
        mean_distances.append(mean_distance)
        print('\rEpisode: {}/{},\tMean Score: {:.2f},\tMean Distance: {:.2f},\tactor_loss: {},\tc1_loss:{},\tc2_loss:{}'
            .format(ep, max_episodes, mean_score, mean_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2))
        FILE = 'record_mean.dat'
        data = [ep, mean_score, mean_distance, mean_loss_actor, mean_loss_c1, mean_loss_c2]
        with open(FILE, "ab") as f:
            pickle.dump(data, f)
env.close()
