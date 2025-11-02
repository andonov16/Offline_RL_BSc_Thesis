import torch
import optuna
import numpy as np
from tqdm import tqdm
from src.tuning.base_objective import BaseObjectiveTorch
from src.architectures import BC


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

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparam_suggestions['lr'],
            weight_decay=hyperparam_suggestions['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max' if self.maximize_score else 'min',
            factor=0.5,
            patience=4
        )

        # early stopping tracking for this trial
        best_trial_score = float('-inf') if self.maximize_score else float('inf')
        epoches_without_improvement = 0
        score = None

        for epoch in tqdm(range(self.max_epochs), desc=f'Trial {trial.number} Epochs'):
            avg_train_loss_per_batch = self.__train_model_single_epoch__(model, optimizer)

            # evaluate the model
            score = self.__evaluate_model_single_epoch__(model)
            scheduler.step(score)
            trial.report(score, epoch)

            # Optuna pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Early stopping: compare with trial-specific best
            improved = (score > best_trial_score) if self.maximize_score else (score < best_trial_score)

            if improved:
                best_trial_score = score
                epoches_without_improvement = 0
            else:
                epoches_without_improvement += 1

            if epoches_without_improvement >= self.early_stopping_criterion_epochs:
                break

        # Update global best model if this trial achieved a new record
        if (self.maximize_score and best_trial_score > self.best_score) or \
        (not self.maximize_score and best_trial_score < self.best_score):
            self.best_score = best_trial_score
            self.__save_best_model__(model=model, model_name=self.model_name)

        del model, optimizer
        torch.cuda.empty_cache()

        return best_trial_score


    def _get_hyperparam_suggestions(self, trial: optuna.Trial) -> dict:
        ss = self.config["search_space"]
        suggestions = {}

        for name, cfg in ss.items():
            # --- Case 1: Fixed parameter (for retraining phase)
            if "value" in cfg:
                suggestions[name] = cfg["value"]
                continue

            # --- Case 2: Normal Optuna-sampled parameter
            param_type = cfg.get("type", "float")

            if param_type == "float":
                # Build kwargs dynamically (avoid passing missing keys)
                kwargs = {
                    "name": name,
                    "low": float(cfg["low"]),
                    "high": float(cfg["high"]),
                }
                if "step" in cfg:
                    kwargs["step"] = float(cfg["step"])
                if "log" in cfg:
                    kwargs["log"] = bool(cfg["log"])

                suggestions[name] = trial.suggest_float(**kwargs)

            elif param_type == "int":
                kwargs = {
                    "name": name,
                    "low": int(cfg["low"]),
                    "high": int(cfg["high"]),
                }
                if "step" in cfg:
                    kwargs["step"] = int(cfg["step"])
                # Optuna’s suggest_int can also take `log`, but you rarely use it
                if "log" in cfg:
                    kwargs["log"] = bool(cfg["log"])

                suggestions[name] = trial.suggest_int(**kwargs)

            else:
                raise ValueError(f"Unsupported parameter type '{param_type}' for '{name}'")

        return suggestions


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