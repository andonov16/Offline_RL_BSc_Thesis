import torch
import optuna

from typing import Tuple
from tqdm import tqdm

from src.tuning.base_objective import BaseObjectiveTorch
from src.Q_network import QNetwork


class DQNBCObjectiveTorch(BaseObjectiveTorch):
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
                 num_features: int = 8,
                 report_every_n_steps: int = 1000,
                 config: dict = ()):
        super(DQNBCObjectiveTorch, self).__init__(train_loader,
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
        self.report_every_n_steps = report_every_n_steps

        self.overall_best_loss = float('inf')
        self.config = config

    def __call__(self, trial: optuna.Trial) -> float:
        # get hyperparameters
        hyperparam_suggestions = self._get_hyperparam_suggestions(trial)

        # define model and optimizer
        online_network = (QNetwork(input_neurons=self.n_features,
                                  hidden_neurons=hyperparam_suggestions['num_hidden_neurons'],
                                  num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
                                  out_neurons=4,
                                  dropout=hyperparam_suggestions['dropout'])
                          .to(self.device))
        target_network = (QNetwork(input_neurons=self.n_features,
                                  hidden_neurons=hyperparam_suggestions['num_hidden_neurons'],
                                  num_hidden_layers=hyperparam_suggestions['num_hidden_layers'],
                                  out_neurons=4,
                                  dropout=hyperparam_suggestions['dropout'])
                          .to(self.device))

        with torch.no_grad():
            target_network.load_state_dict(online_network.state_dict())

        optimizer = torch.optim.Adam(
            online_network.parameters(),
            lr=hyperparam_suggestions['lr']
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )

        # Track the best evaluation loss for early stopping
        curr_best_loss = float('inf')
        iterations_without_improvement = 0

        replay_buffer = iter(self.train_loader)

        UPDATE_EVERY = 1000
        PBAR_EVERY = 1000
        pbar = tqdm(total=self.max_num_training_iters,
                    desc=f'Trial {trial.number}',
                    miniters=UPDATE_EVERY,
                    mininterval=1.0)

        for iteration in range(self.max_num_training_iters):
            # get a random mini-batch
            try:
                mini_batch = next(replay_buffer)
            except StopIteration:   # restart when finished
                replay_buffer = iter(self.train_loader)
                mini_batch = next(replay_buffer)

            # training step
            update_target_network_flag = iteration % hyperparam_suggestions['target_update_rate'] == 0
            train_loss = self._train_network_for_single_iteration(
                online_network=online_network,
                target_network=target_network,
                q_optimizer=optimizer,
                mini_batch=mini_batch,
                threshold=hyperparam_suggestions['theta'],
                update_target_network=update_target_network_flag
            )

            if iteration % PBAR_EVERY == 0:
                pbar.update(PBAR_EVERY)
                pbar.set_postfix(loss=float(train_loss))

            if iteration % 1000 == 0:
                scheduler.step(train_loss)

            # report to Optuna (for pruning)
            if iteration % self.report_every_n_steps == 0:
                trial.report(train_loss, iteration)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # early stopping based on eval loss
            if train_loss < curr_best_loss:
                curr_best_loss = train_loss
                iterations_without_improvement = 0
                # save best model during trial
                if self.overall_best_loss > curr_best_loss:
                    self.best_model = online_network
                    self._save_best_model(
                        model=online_network,
                        model_name=self.model_name
                    )
                    self.overall_best_loss = curr_best_loss
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= self.early_stopping_criterion_iters:
                break

        # Clean up
        del online_network, target_network, optimizer
        torch.cuda.empty_cache()
        pbar.close()

        # Return the evaluation loss as the trial score
        return curr_best_loss

    def _get_hyperparam_suggestions(self, trial: optuna.Trial) -> dict:
        search_space = self.config['search_space']
        suggestions = {}

        for name, cfg in search_space.items():
            if cfg.get('is_constant', False):
                suggestions[name] = cfg['value']
                continue

            param_type = cfg['type']

            if param_type == 'float':
                suggest_kwargs = {
                    'name': name,
                    'low': float(cfg['low']),
                    'high': float(cfg['high']),
                }

                if 'step' in cfg:
                    suggest_kwargs['step'] = float(cfg['step'])

                if 'log' in cfg:
                    suggest_kwargs['log'] = bool(cfg['log'])

                suggestions[name] = trial.suggest_float(**suggest_kwargs)

            elif param_type == 'int':
                suggest_kwargs = {
                    'name': name,
                    'low': int(cfg['low']),
                    'high': int(cfg['high']),
                }

                if 'step' in cfg:
                    suggest_kwargs['step'] = int(cfg['step'])

                if 'log' in cfg:
                    suggest_kwargs['log'] = bool(cfg['log'])

                suggestions[name] = trial.suggest_int(**suggest_kwargs)

            else:
                raise ValueError(f'Unsupported parameter type "{param_type}" for "{name}"')

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
                                            threshold: float = 0, # theta = 0 -> returns Q-learning
                                            update_target_network: bool = False
                                        ) -> float:
        # Implementation of algorithm 1 from https://arxiv.org/pdf/1910.01708

        online_network.train()
        target_network.eval()
        self.generative_model.eval()  # frozen BC model

        states, actions, rewards, next_states, dones = mini_batch

        # compute next actions with BC filtering
        with torch.no_grad():
            # behavior model probabilities
            gen_probs = self.generative_model(next_states)  # [batch, num_actions]
            max_gen_probs, _ = gen_probs.max(dim=1, keepdim=True)
            mask = (gen_probs / max_gen_probs.clamp(min=1e-8)) >=  threshold

            # online Q-values for action selection (Double DQN style)
            q_next_online = online_network(next_states)  # [batch, num_actions]
            masked_q_next = q_next_online.masked_fill(~mask, float('-inf'))
            next_actions = masked_q_next.argmax(dim=1)  # [batch]

            # target Q-values from target network
            q_next_target = target_network(next_states)
            target_q_values = q_next_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        current_q_values = online_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # compute targets
        targets = rewards + self.gamma * target_q_values * (1 - dones)
        q_loss = torch.nn.functional.smooth_l1_loss(current_q_values, targets.detach())

        # backpropagate
        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        # update target network if needed
        if update_target_network:
            with torch.no_grad():
                target_network.load_state_dict(online_network.state_dict())

        return q_loss.detach()