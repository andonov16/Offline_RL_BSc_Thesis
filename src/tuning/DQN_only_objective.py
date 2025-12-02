import torch
import optuna

from typing import Tuple
from tqdm import tqdm

from src.tuning.base_objective import BaseObjectiveTorch
from src.Q_network import QNetwork


class DQNOnlyObjectiveTorch(BaseObjectiveTorch):
    def __init__(self,
                 train_loader: torch.utils.data.DataLoader,
                 device: torch.device,
                 dataset_name='final_policy',
                 model_name: str = 'DQN_only_agent_raw',
                 model_dir: str = '../models',
                 logs_dir: str = '../logs',
                 max_num_training_iters: int = 1000000,
                 early_stopping_criterion_iters: int = 50000,
                 gamma: float = 0.99,
                 generative_model: torch.nn.Module = None,
                 num_features: int = 9,
                 config: dict = ()):
        super(DQNOnlyObjectiveTorch, self).__init__(train_loader,
                                                   device,
                                                   model_dir,
                                                   logs_dir)
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.n_features = num_features

        self.gamma = gamma
        self.max_num_training_iters = max_num_training_iters
        self.early_stopping_criterion_iters = early_stopping_criterion_iters
        self.generative_model = generative_model

        self.overall_best_loss = float('inf')
        self.config = config

    def __call__(self, trial: optuna.Trial) -> float:
        # get hyperparameters
        hyperparam_suggestions = self._get_hyperparam_suggestions(trial)

        # define model and optimizer
        online_network = (QNetwork(input_neurons=self.n_features,
                                  hidden_neurons=hyperparam_suggestions['hidden_neurons'],
                                  num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
                                  out_neurons=4,
                                  dropout=hyperparam_suggestions['dropout'])
                          .to(self.device))
        target_network = (QNetwork(input_neurons=self.n_features,
                                  hidden_neurons=hyperparam_suggestions['hidden_neurons'],
                                  num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
                                  out_neurons=4,
                                  dropout=hyperparam_suggestions['dropout'])
                          .to(self.device))

        optimizer = torch.optim.Adam(
            online_network.parameters(),
            lr=hyperparam_suggestions['lr'],
            weight_decay=hyperparam_suggestions['weight_decay']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=int(self.early_stopping_criterion_iters*0.25)
        )

        # Track the best evaluation loss for early stopping
        curr_best_loss = float('inf')
        iterations_without_improvement = 0

        replay_buffer = iter(self.train_loader)

        for iteration in tqdm(range(self.max_num_training_iters), desc=f'Trial {trial.number} Iterations'):
            # get a random mini-batch
            try:
                mini_batch = next(replay_buffer)
            except StopIteration:   # restart when finished
                replay_buffer = iter(self.train_loader)
                mini_batch = next(replay_buffer)

            # training step
            train_loss = self._train_network_for_single_iteration(
                online_network=online_network,
                target_network=target_network,
                q_optimizer=optimizer,
                mini_batch=mini_batch,
                threshold=0,
            )
            scheduler.step(train_loss)

            # report to Optuna (for pruning)
            trial.report(train_loss, iteration)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # early stopping based on eval loss
            if train_loss < curr_best_loss:
                curr_best_eval_loss = train_loss
                iterations_without_improvement = 0
                # save best model during trial
                if self.overall_best_loss > curr_best_eval_loss:
                    self.best_model = online_network
                    self.__save_best_model__(model=online_network, model_name=self.model_name)
                    self.overall_best_loss = curr_best_eval_loss
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.early_stopping_criterion_iters:
                break

        # Clean up
        del online_network, target_network, optimizer
        torch.cuda.empty_cache()

        # Return the evaluation loss as the trial score
        return curr_best_loss

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

    def _train_network_for_single_iteration(self,
                                            online_network: torch.nn.Module,
                                            target_network: torch.nn.Module,
                                            q_optimizer: torch.optim.Optimizer,
                                            mini_batch: Tuple[
                                                torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor],
                                            threshold: float = 0,
                                            update_target_network: bool = False
                                        ) -> float:
        # Implementation of algorithm 1 from https://arxiv.org/pdf/1910.01708
        online_network.train()
        target_network.eval()
        self.generative_model.eval()

        states, actions, rewards, next_states, dones = mini_batch

        # Line 5: action selection with threshold (in this case threshold = 0 -> only BC values will be totally ignored
        with torch.no_grad():
            gen_probs = self.generative_model(next_states)  # [batch, num_actions]
            q_values_next = online_network(next_states)  # [batch, num_actions]

            max_gen_probs, _ = gen_probs.max(dim=1, keepdim=True)
            mask = (gen_probs / max_gen_probs) > threshold

            # Masked Q-values: set invalid actions to -inf
            masked_q_values = q_values_next.masked_fill(~mask, float('-inf'))

            # Select best action under threshold condition
            next_actions = masked_q_values.argmax(dim=1)

        # Line 6: online Q-network update
        online_q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        target_q_values = target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + self.gamma * target_q_values * (1 - dones)
        # using the Hubert loss instead of plain MSE as suggested in the paper
        q_loss = torch.nn.functional.smooth_l1_loss(online_q_values, targets.detach())

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # Line 8: target Q-network update (if necessary)
        if update_target_network:
            target_network.load_state_dict(online_network.state_dict())

        return q_loss.item()