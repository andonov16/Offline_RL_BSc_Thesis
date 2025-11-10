import torch
import os
import gymnasium as gym
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from gymnasium import register
from tqdm import tqdm


def BC_evaluate_model_in_live_env(env_test_params = dict,
                               model_name: str = 'Replay Buffer',
                               norm_technique: torch.nn.Module = None,
                               model =  torch.jit.load('../models/replay_buffer/BC_standard_refined.pt')) -> np.array:
    rewards = []

    model.eval()
    log_subfolder = f'../logs/{model_name}/BC_live_env_performance_test/'
    tensorboard_log_subfolder = os.path.join(log_subfolder, 'tensorboard')
    if not os.path.exists(log_subfolder):
        os.makedirs(log_subfolder)
    if not os.path.exists(tensorboard_log_subfolder):
        os.makedirs(tensorboard_log_subfolder)
    log_writer = SummaryWriter(log_dir=tensorboard_log_subfolder)

    register(
        id='LunarLander-v2',
        entry_point='gymnasium.envs.box2d:LunarLander',
        max_episode_steps=1000,
        reward_threshold=200,
    )

    env = gym.make('LunarLander-v2')
    env.action_space.seed(env_test_params['seed'])

    for episode in tqdm(range(env_test_params['num_episodes'])):
        state, info = env.reset()
        done = False
        total_reward = 0
        reward = 0

        while not done:
            features = torch.tensor(state, dtype=torch.float32)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            model_input = torch.cat((features, reward_tensor), dim=0)

            if model_input is not None:
                model_input = norm_technique(model_input)

            model_output = model(model_input)
            action = torch.argmax(torch.softmax(model_output, dim=-1)).item()
            next_state, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            done = terminated or truncated
            state = next_state

        log_writer.add_scalar('Episode Reward', total_reward, episode)
        rewards.append(total_reward)

    log_writer.add_scalar('Reward (Mean)', np.mean(rewards), 0)
    log_writer.add_scalar('Reward (std)', np.std(rewards), 0)

    log_writer.flush()
    log_writer.close()
    env.close()

    return np.array(rewards)