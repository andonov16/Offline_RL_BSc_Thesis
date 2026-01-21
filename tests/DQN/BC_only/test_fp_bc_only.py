import torch
from src.utils.live_env_testing import evaluate_model_in_live_env_UI_available


if __name__ == '__main__':
    model = torch.jit.load('../../../models/DQN/final_policy/BC_only_standard.pt')
    norm_technique = torch.jit.load('../../../models/BC/final_policy/normalization/standard_normalization.pt')

    evaluate_model_in_live_env_UI_available(
        model_to_evaluate=model,
        norm_technique=norm_technique,
    )