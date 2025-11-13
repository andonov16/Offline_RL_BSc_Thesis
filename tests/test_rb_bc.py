# run this file to see how the BC agent trained on the replay buffer dataset performs in the live environment

import torch
import gymnasium as gym
from gymnasium.envs.registration import register


from src.utils.config_managing import *
from src.behaviour_cloning import BC

BC_model = torch.jit.load('../models/replay_buffer/BC/BC_standard_refined.pt')
norm_technique = torch.jit.load('../models/replay_buffer/BC/normalization/standard_normalization.pt')

# create a register and an env object (as shown in the notebook provided with the task)
register(
    id='LunarLander-v2',
    entry_point='gymnasium.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

# Separate env for evaluation
env = gym.make(id='LunarLander-v2', render_mode='human')

# run the environment visually to see the agent behaviour
episode = 0
while True:
    state, info = env.reset()
    done = False
    total_reward = 0
    reward = 0

    while not done:
        # convert the state from np to tensor to pass it through the BC model
        features = torch.tensor(state, dtype=torch.float32)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        model_input = torch.cat((features, reward_tensor), dim=0)

        if model_input is not None:
            model_input = norm_technique(model_input)

        model_output = BC_model(model_input)
        action = torch.argmax(torch.softmax(model_output, dim=-1)).item()

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        state = next_state

    print(f'Episode {episode + 1}: Total Reward = {total_reward}')
    episode += 1
env.close()
