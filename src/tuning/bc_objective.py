import torch
import optuna
import numpy as np
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

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
        super(BCObjectiveTorch, self).__init__(train_loader=train_loader,
                                               device=device,
                                               model_dir=model_dir,
                                               logs_dir=logs_dir,
                                               config=config)
        self.eval_loader = eval_loader
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.max_epochs = max_epochs
        self.early_stopping_criterion_epochs = max_epochs_without_improvement
        self.n_features = num_features

        self.overall_best_loss = float('inf')

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=torch.unique(train_loader.dataset.actions).cpu().detach().numpy(),
            y=train_loader.dataset.actions.cpu().detach().numpy()
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        self.loss_func = torch.nn.CrossEntropyLoss(weight=class_weights.to(self.device))

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
            lr=hyperparam_suggestions['lr']
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
            train_loss = self._train_model_single_epoch(model, optimizer)
            train_losses.append(train_loss)

            # evaluation step (compute eval loss)
            # returns eval loss
            eval_loss = self._evaluate_model_single_epoch(model)
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
                    self._save_best_model(model=model, model_name=self.model_name)
                    self.overall_best_loss = curr_best_eval_loss
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_criterion_epochs:
                break

        # Save train and eval losses in trial user attributes
        trial.set_user_attr('train_losses', train_losses)
        trial.set_user_attr('eval_losses', eval_losses)

        # Clean up
        del model, optimizer
        torch.cuda.empty_cache()

        # Return the evaluation loss as the trial score
        return curr_best_eval_loss

    def _train_model_single_epoch(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> float:
        model.train()

        scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.device.type == 'cuda')

        total_loss = 0.0
        for X, Y_true in self.train_loader:
            optimizer.zero_grad(set_to_none=True)
            X=X.to(self.device)
            Y_true=Y_true.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type == 'cuda'):
                preds = model(X)
                curr_loss = self.loss_func(preds, Y_true)

            scaler.scale(curr_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += curr_loss.item()
            del preds
        return total_loss / len(self.train_loader)
        
    def _evaluate_model_single_epoch(self, model: torch.nn.Module) -> float:
        model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X, Y_true in self.eval_loader:
                X=X.to(self.device)
                Y_true=Y_true.to(self.device)

                Y_pred = model(X)
                # Compute loss
                total_loss += self.loss_func(Y_pred, Y_true).item()
        return total_loss / len(self.eval_loader)
