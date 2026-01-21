# run this file to see how the BC agent trained on the replay buffer dataset performs in the live environment
import torch
from src.utils.live_env_testing import evaluate_model_in_live_env_UI_available


if __name__ == '__main__':
    BC_model = torch.jit.load('../../models/BC/final_policy/BC_raw.pt')
    norm_technique = None

    evaluate_model_in_live_env_UI_available(
        model_to_evaluate=BC_model,
        norm_technique=norm_technique,
    )