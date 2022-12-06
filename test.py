import gym
from TD3 import TD3
from PIL import Image
import numpy as np
from typing import Union
from gym import spaces
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

gym.logger.set_level(40)
env_name = "BipedalWalker-v3"
random_seed = 0
n_episodes = 100
lr = 0.002
max_timesteps = 2000
render = True
save_gif = False

filename = "TD3_{}_{}".format(env_name, random_seed)
filename += '_solved'
directory = "./preTrained/".format(env_name)
episode = 898

env = CustomWrapper(gym.make(env_name),  min_action = -0.5,  max_action = 0.5)
# state_dim = env.observation_space.shape[0]
state_dim = 20
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

policy = TD3(lr, state_dim, action_dim, max_action)

policy.load_actor(directory, filename, episode)

scores = []

for ep in range(1, n_episodes+1):
    ep_reward = 0
    state = env.reset()
    for t in range(max_timesteps):
        action = policy.select_action(state[:20])
        state, reward, done, _ = env.step(action)
        ep_reward += reward
        if render:
            env.render()
            if save_gif:
                 img = env.render(mode = 'rgb_array')
                 img = Image.fromarray(img)
                 img.save('./gif/{}.jpg'.format(t))
        if done:
            break
    scores.append(ep_reward)
    print('Episode: {}\tReward: {}'.format(ep, int(ep_reward)))
    env.close()


print("Score media", np.mean(scores))
    
