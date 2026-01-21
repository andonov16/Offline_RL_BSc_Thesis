import torch
from src.utils.live_env_testing import evaluate_model_in_live_env_UI_available


if __name__ == '__main__':
    model = torch.jit.load('../../../models/DQN/replay_buffer/BC_only_raw.pt')
    norm_technique = None

    evaluate_model_in_live_env_UI_available(
        model_to_evaluate=model,
        norm_technique=norm_technique,
    )