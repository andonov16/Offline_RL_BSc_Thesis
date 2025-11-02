import torch
import optuna
import os
import copy
from sklearn.utils.class_weight import compute_class_weight


class BaseObjectiveTorch:
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 model_dir: str = '../models',
                 logs_dir: str = '../logs',
                 scoring_fn: callable = None,
                 maximize_score: bool = True):
        self.train_loader = train_loader
        self.model_dir = model_dir
        self.log_dir = logs_dir
        self.scoring_fn = scoring_fn
        self.maximize_score = maximize_score
        self.device = device

        self.best_model = None
        if maximize_score:
            self.best_score = -torch.inf
        else:
            self.best_score = torch.inf

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=torch.unique(train_loader.dataset.actions).cpu().detach().numpy(),
            y=train_loader.dataset.actions.cpu().detach().numpy()
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))


    def __call__(self, trial: optuna.Trial) -> float:
        raise NotImplemented()


    def __save_best_model__(self, model: torch.nn.Module, model_name: str = 'best_model') -> None:
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f'{model_name}.pt')

        model_cpu = copy.deepcopy(model).to('cpu').eval()

        scripted_model = torch.jit.script(model_cpu)
        scripted_model.save(model_path)