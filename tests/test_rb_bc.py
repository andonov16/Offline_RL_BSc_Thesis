# run this file to see how the BC agent trained on the replay buffer dataset performs in the live environment

import torch
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register


from src.utils.config_managing import *
from src.behavior_cloning import BC

env_test_params = load_env_test_config_file()
with open(f"../logs/rb_bc/best_hyperparams.txt", "r") as file:
    hyperparams_dict_text = file.readline().strip()
    hyperparams_dict_text = hyperparams_dict_text[22:].replace("'", '"')
    best_model_hyperparams = json.loads(hyperparams_dict_text)

# create a BC object with the same structure as the best performing model and
# load the best model`s weights
BC_model = BC(input_neurons=best_model_hyperparams['input_neurons'],
              hidden_neurons=best_model_hyperparams['hidden_neurons'],
              num_hidden_layers=best_model_hyperparams['num_hidden_layers'],
              out_neurons=best_model_hyperparams['out_neurons'],
              activation_function=torch.nn.ReLU())
BC_model.load_state_dict(torch.load(f"../models/rb_BC_best_model.pth"))
BC_model.eval()


# create a register and an env object (as shown in the notebook provided with the task)
register(
    id="LunarLander-v2",
    entry_point="gymnasium.envs.box2d:LunarLander",
    max_episode_steps=1000,
    reward_threshold=200,
)

# Separate env for evaluation
env = gym.make("LunarLander-v2", render_mode='human')
env.action_space.seed(env_test_params['seed'])

# run the environment visually to see the agent behaviour
for episode in range(env_test_params['num_episodes']):
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # convert the state from np to tensor to pass it through the BC model
        model_output = BC_model(torch.tensor(state))
        action = np.argmax(model_output.detach().numpy())
        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        done = terminated or truncated
        state = next_state

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

env.close()