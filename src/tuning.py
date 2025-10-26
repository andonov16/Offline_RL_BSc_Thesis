import optuna
import copy
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

from src.architectures import BC


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


class BCObjectiveTorch(BaseObjectiveTorch):
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 eval_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 dataset_name='final_policy',
                 model_name: str = 'BC_agent_raw',
                 model_dir: str = '../models',
                 logs_dir: str = '../logs',
                 scoring_fn: callable = None,
                 maximize_score: bool = True,
                 max_epochs: int = 100,
                 max_epochs_without_improvement: int = 7,
                 num_features: int = 9,
                 config: dict = ()):
        super(BCObjectiveTorch, self).__init__(train_loader,
                                               device,
                                               model_dir,
                                               logs_dir,
                                               scoring_fn,
                                               maximize_score)
        self.eval_loader = eval_loader
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.early_stopping_criterion_epochs = max_epochs_without_improvement
        self.n_features = num_features

        self.config = config

    def __call__(self, trial: optuna.Trial) -> float:
        # define suggestions
        hyperparam_suggestions = self._get_hyperparam_suggestions(trial)

        # define model and optimizer
        model = BC(input_neurons=self.n_features,
                   hidden_neurons=hyperparam_suggestions['num_hidden_neurons'],
                   num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
                   out_neurons=4,
                   dropout=hyperparam_suggestions['dropout'])
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyperparam_suggestions['lr'],
                                     weight_decay=hyperparam_suggestions['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max' if self.maximize_score else 'min',
            factor=0.5,
            patience=4
        )

        # train the model for the given number of epochs (+ early stopping logic)
        epoches_without_improvement = 0
        score = None
        for epoch in tqdm(range(self.max_epochs), desc='Max Epochs'):
            avg_train_loss_per_batch = self.__train_model_single_epoch__(model, optimizer)

            # evaluate the model
            score = self.__evaluate_model_single_epoch__(model)
            scheduler.step(score)
            trial.report(score, epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # save it sa best if the model`s score is better than the current best one
            if self.maximize_score and score >= self.best_score:
                self.best_score = score
                epoches_without_improvement = 0
                self.__save_best_model__(model=model, model_name=self.model_name)
            elif not self.maximize_score and score <= self.best_score:
                self.best_score = score
                epoches_without_improvement = 0
            else:
                epoches_without_improvement += 1
                if epoches_without_improvement >= self.early_stopping_criterion_epochs:
                    break

        del model, optimizer
        torch.cuda.empty_cache()
        return score

    def _get_hyperparam_suggestions(self, trial: optuna.Trial) -> dict:
        ss = self.config["search_space"]

        dropout = trial.suggest_float(
            name="dropout", low=ss["dropout"]["low"], high=ss["dropout"]["high"]
        )
        lr = trial.suggest_float(
            name="lr", low=ss["lr"]["low"], high=ss["lr"]["high"], log=True
        )
        num_hidden_neurons = trial.suggest_int(
            name="num_hidden_neurons",
            low=ss["num_hidden_neurons"]["low"],
            high=ss["num_hidden_neurons"]["high"],
            step=ss["num_hidden_neurons"].get("step", 1)
        )
        num_hidden_layers = trial.suggest_int(
            name="num_hidden_layers",
            low=ss["num_hidden_layers"]["low"],
            high=ss["num_hidden_layers"]["high"]
        )
        weight_decay = trial.suggest_float(
            name="weight_decay",
            low=ss["weight_decay"]["low"],
            high=ss["weight_decay"]["high"],
            log=True
        )

        result = {
            "dropout": dropout,
            "lr": lr,
            "num_hidden_neurons": num_hidden_neurons,
            "num_hidden_layers": num_hidden_layers,
            "weight_decay": weight_decay,
        }
        return result

    def __train_model_single_epoch__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        losses = []

        scaler = torch.amp.GradScaler(device=self.device, enabled=self.device.type == "cuda")

        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            for X, Y_true in self.train_loader:
                X, Y_true = X.to(self.device, non_blocking=True), Y_true.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.device.type == "cuda"):
                    preds = model(X)
                    loss = self.loss_func(preds, Y_true)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                del X, Y_true, preds, loss

        torch.cuda.empty_cache()
        return float(np.mean(losses))

    def __evaluate_model_single_epoch__(self, model: torch.nn.Module) -> float:
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for X, Y_true in self.eval_loader:
                X, Y_true = X.to(self.device), Y_true.to(self.device)
                Y_pred = model(X)
                preds = torch.argmax(Y_pred, dim=1)
                all_preds.extend(preds.detach().cpu().numpy())
                all_labels.extend(Y_true.detach().cpu().numpy())

                del X, Y_true, Y_pred, preds

            torch.cuda.empty_cache()
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            score = self.scoring_fn(all_labels, all_preds)

        del all_preds, all_labels
        torch.cuda.empty_cache()
        return score