import torch
import optuna
import numpy as np
from tqdm import tqdm
from src.tuning.base_objective import BaseObjectiveTorch
from src.behaviour_cloning import BC


class BCObjectiveTorch(BaseObjectiveTorch):
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 eval_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 dataset_name='final_policy',
                 model_name: str = 'BC_agent_raw',
                 model_dir: str = '../models',
                 logs_dir: str = '../logs',
                 max_epochs: int = 100,
                 max_epochs_without_improvement: int = 7,
                 num_features: int = 9,
                 config: dict = ()):
        super(BCObjectiveTorch, self).__init__(train_loader,
                                               device,
                                               model_dir,
                                               logs_dir)
        self.eval_loader = eval_loader
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.early_stopping_criterion_epochs = max_epochs_without_improvement
        self.n_features = num_features

        self.overall_best_loss = float('inf')
        self.config = config

    def __call__(self, trial: optuna.Trial) -> float:
        # get hyperparameters
        hyperparam_suggestions = self._get_hyperparam_suggestions(trial)

        # define model and optimizer
        model = BC(
            input_neurons=self.n_features,
            hidden_neurons=hyperparam_suggestions['num_hidden_neurons'],
            num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
            out_neurons=4,
            dropout=hyperparam_suggestions['dropout']
        )
        model.to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparam_suggestions['lr'],
            weight_decay=hyperparam_suggestions['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',  # because lower loss is better
            factor=0.5,
            patience=4
        )

        # Track best evaluation loss for early stopping
        curr_best_eval_loss = float('inf')
        epochs_without_improvement = 0

        # Lists to store losses per epoch
        train_losses = []
        eval_losses = []

        for epoch in tqdm(range(self.max_epochs), desc=f'Trial {trial.number} Epochs'):
            # training step
            train_loss = self.__train_model_single_epoch__(model, optimizer)
            train_losses.append(train_loss)

            # evaluation step (compute eval loss)
            # returns eval loss
            eval_loss = self.__evaluate_model_single_epoch__(model) 
            eval_losses.append(eval_loss)

            # scheduler step
            scheduler.step(eval_loss)

            # report to Optuna (for pruning)
            trial.report(eval_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # early stopping based on eval loss
            if eval_loss < curr_best_eval_loss:
                curr_best_eval_loss = eval_loss
                epochs_without_improvement = 0
                # save best model during trial
                if self.overall_best_loss > curr_best_eval_loss:
                    self.best_model = model
                    self.__save_best_model__(model=model, model_name=self.model_name)
                    self.overall_best_loss = curr_best_eval_loss
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_criterion_epochs:
                break

        self.__evaluate_model_single_epoch_accuracy__(self.best_model)

        # Save train and eval losses in trial user attributes
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('eval_losses', eval_losses)

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        # Return the evaluation loss as the trial score
        return curr_best_eval_loss

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


    def __train_model_single_epoch__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()
        losses = []

        scaler = torch.amp.GradScaler(device=self.device, enabled=self.device.type == 'cuda')

        with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
            for X, Y_true in self.train_loader:
                X, Y_true = X.to(self.device, non_blocking=True), Y_true.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
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
        losses = []

        with torch.no_grad():
            for X, Y_true in self.eval_loader:
                X, Y_true = X.to(self.device), Y_true.to(self.device)
                Y_pred = model(X)
                # Compute loss
                loss = self.loss_func(Y_pred, Y_true)
                losses.append(loss.item())
                del X, Y_true, Y_pred, loss

        torch.cuda.empty_cache()

        # Compute average validation loss
        avg_loss = float(np.mean(losses))
        del losses
        torch.cuda.empty_cache()
        return avg_loss

    def __evaluate_model_single_epoch_accuracy__(self, model: torch.nn.Module) -> float:
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for X, Y_true in self.eval_loader:
                X, Y_true = X.to(self.device), Y_true.to(self.device)
                Y_pred = model(X)

                # Get predicted classes
                Y_pred_classes = torch.argmax(Y_pred, dim=1)

                # Count correct predictions
                correct += (Y_pred_classes == Y_true).sum().item()
                total += Y_true.size(0)

                del X, Y_true, Y_pred, Y_pred_classes

        torch.cuda.empty_cache()

        # Compute accuracy
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        print(f'Validation Accuracy: {accuracy:.2f}%')

        return accuracy