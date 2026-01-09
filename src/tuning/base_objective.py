import torch
import optuna
import os
import copy


class BaseObjectiveTorch:
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 model_dir: str = '../models',
                 logs_dir: str = '../logs',
                 maximize_score: bool = False,
                 config: dict = None,):
        self.train_loader = train_loader
        self.model_dir = model_dir
        self.log_dir = logs_dir
        self.maximize_score = maximize_score
        self.device = device
        self.config = config

        self.best_model = None
        if maximize_score:
            self.best_score = -torch.inf
        else:
            self.best_score = torch.inf


    def __call__(self, trial: optuna.Trial) -> float:
        raise NotImplemented()


    def _get_hyperparam_suggestions(self, trial: optuna.Trial) -> dict:
        ss = self.config['search_space']
        suggestions = {}

        for name, cfg in ss.items():
            # Case 1: Fixed parameter (for retraining phase)
            if 'value' in cfg:
                suggestions[name] = cfg['value']
                continue

            # Case 2: Normal Optuna-sampled parameter
            param_type = cfg.get('type', 'float')

            if param_type == 'float':
                # Build kwargs dynamically (avoid passing missing keys)
                kwargs = {
                    'name': name,
                    'low': float(cfg['low']),
                    'high': float(cfg['high']),
                }
                if 'step' in cfg:
                    kwargs['step'] = float(cfg['step'])
                if 'log' in cfg:
                    kwargs['log'] = bool(cfg['log'])

                suggestions[name] = trial.suggest_float(**kwargs)

            elif param_type == 'int':
                kwargs = {
                    'name': name,
                    'low': int(cfg['low']),
                    'high': int(cfg['high']),
                }
                if 'step' in cfg:
                    kwargs['step'] = int(cfg['step'])

                if 'log' in cfg:
                    kwargs['log'] = bool(cfg['log'])

                suggestions[name] = trial.suggest_int(**kwargs)

            else:
                raise ValueError(f'Unsupported parameter type {param_type} for {name}')

        return suggestions

    def _save_best_model(self, model: torch.nn.Module, model_name: str = 'best_model') -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'{model_name}.pt')

        model_cpu = copy.deepcopy(model).to('cpu').eval()

        scripted_model = torch.jit.script(model_cpu)
        scripted_model.save(model_path)